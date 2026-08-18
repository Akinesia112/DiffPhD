/////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2022,
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "SeSchwarzPreconditioner.h"

#include "SeIntrinsic.h"
#if !defined(WIN32) && (defined(__AVX2__) || defined(__FMA__))
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstdio>
#include <unordered_map>


SE_USING_NAMESPACE

// entry point 1
void SeSchwarzPreconditioner::AllocatePrecoditioner(int numVerts, int numEdges, int numFaces)
{
	m_numVerts = numVerts;
	m_numEdges = numEdges;
	m_numFaces = numFaces;

	if (m_frameIndex == 0)
	{
		DoAlllocation();
	}

	if (m_frameIndex % 17 != 0)
	{
		return;
	}

	//==== Reorder 

	ComputeTotalAABB();

	// [對應論文 Section 5.1: Domain Partitioning]
    // "We choose to sort the nodes by their Morton codes first." 
    // 進行空間排序以優化記憶體局部性 (Locality) 並作為分群基礎。
	SpaceSort();

	ComputeInverseMapper();

	MapHessianTable();

	m_frameIndex++;
}

// entry point 2
void SeSchwarzPreconditioner::PreparePreconditioner
(
	const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges,
	const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, 
	unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts
)
{
	m_mappedNeighborsRemain = m_mappedNeighbors;
	m_mappedNeighborsNumRemain = m_mappedNeighborsNum;

	//==== before reordering

	PrepareCollisionStencils(efSets, eeSets, vfSets, efCounts, eeCounts, vfCounts);

	//====
	// [對應論文 Section 5.2: Coarse Space Construction]
    // "construct a Nicolaides' coarse space by grouping nodes... into supernodes." [cite: 395]
    // 建立多層級結構 (Hierarchy)。
	ReorderRealtime();
	
	//====
	
	const int blockNum = m_totalSz / 32;
	Utility::MemsetZero(m_hessian32.Ptr(), m_hessian32.Size());
	Utility::MemsetZero(m_additionalHessian32);
	
	//====

	// [對應論文 Section 6: Matrix Precomputation]
    // 準備每個子區域 (Domain) 的矩陣與其反矩陣。
	PrepareCollisionHessian(); //3

	// [對應論文 Algorithm 1: Matrix Precomputation]
    // 步驟 1: 組裝子矩陣 A_{d,(l)}
	PrepareHessian(diagonal, csrOffDiagonals, csrRanges);

	// [對應論文 Algorithm 1 & Section 6.2]
    // 步驟 2 & 3: 計算子矩陣的逆 M_{d,(l)}^{-1}
	LDLtInverse512();
}

// =========================================================
// SECTION 7: RUNTIME PRECONDITIONING
// =========================================================

// [對應論文 Equation 5: The MAS Preconditioner]
// 公式：M_{MAS}^{-1} = M_{(0)}^{-1} + Sum( C_{(l)}^T M_{(l)}^{-1} C_{(l)} )
// 這是執行時期的主要入口函數。
// entry point 3
void SeSchwarzPreconditioner::Preconditioning(SeVec3fSimd* z, const SeVec3fSimd* residual, int dim)
{
	Utility::MemsetZero(m_mappedZ); 
	Utility::MemsetZero(m_mappedR);

	// 步驟 1: 限制 (Restriction) / 分配
	// [對應論文 Section 4 & 7] "Distributing the input x to every level"
	// 實際上是在計算 C_{(l)} * r
	BuildResidualHierarchy(residual);

	// 步驟 2: 局部求解 (Local Solving)
	// [對應論文 Section 7.1] "Performing the matrix-vector product in each domain"
	// 計算 y_(l) = M_{d,(l)}^{-1} * x_(l)
	SchwarzLocalXSym();

	// 步驟 3: 插值 (Prolongation) 與 加總
	// [對應論文 Equation 5 的求和部分]
	// 將各層級的結果加總：z = Sum( y_(l) )
	CollectFinalZ(z);
}

// [對應論文 Section 7.1: Symmetric Matrix-Vector Multiplication]
// 實現論文 Fig 12 與 Fig 13 描述的優化版對稱矩陣向量乘法。
void SeSchwarzPreconditioner::ComputeLevelNums(int bankSize)
{
	constexpr float SizeRatio = 1.5f;

	//====

	int totalSz = 0;

	int nLevel = 1;
	int levelSz = (m_numVerts + bankSize - 1) / bankSize * bankSize;
	totalSz += levelSz;

	while (levelSz > 32)
	{
		levelSz /= 32;

		nLevel++;
		levelSz = (levelSz + bankSize - 1) / bankSize * bankSize;
		totalSz += levelSz;
	}

	m_numLevel = nLevel;
	m_totalSz = totalSz * SizeRatio;
}

void SeSchwarzPreconditioner::DoAlllocation()
{
	constexpr int bankSize = 32;	//==== size of diagonal block

	ComputeLevelNums(bankSize);

	//====

	m_MapperSortedGetOriginal.resize(m_numVerts);
	m_mapperOriginalGetSorted.resize(m_numVerts);
	m_mortonCode.resize(m_numVerts);

	m_hessian32.Resize(bankSize, m_totalSz);
	m_mappedZ.resize(m_totalSz);
	m_mappedR.resize(m_totalSz);

	m_mappedPos.resize(m_numVerts);

	m_denseLevel.resize(m_numVerts);
	m_goingNext.resize(m_numLevel * m_numVerts);
	m_coarseTables.resize(m_numVerts);

	m_prefixOrignal.resize(m_numVerts);
	m_fineConnectMask.resize(m_numVerts);
	m_nextConnectMsk.resize(m_numVerts);
	m_nextPrefix.resize(m_numVerts);


	int triSz = (1 + 96) * 96 / 2 + 16 * 3;
	const int blockNum = m_totalSz / 32;
	m_invSymR.resize(triSz * blockNum);
	m_additionalHessian32.resize(m_totalSz + 1);

	//====

	m_levelSize.resize(m_numLevel + 1);
	m_CoarseSpaceTables.Resize(m_numLevel, m_numVerts);

	int maxNeighbours = 0;

	for (int vId = 0; vId < m_numVerts; ++vId)
	{
		maxNeighbours = Math::Max(m_neighbours->Size(vId) + 1, maxNeighbours);
	}

	m_mappedNeighborsNum.resize(m_numVerts);
	m_mappedNeighborsNumRemain.resize(m_numVerts);
	m_mappedNeighbors.Resize(maxNeighbours, m_numVerts);
	m_mappedNeighborsRemain.Resize(maxNeighbours, m_numVerts);

	const int maxCollisionPerVert = 32;
	m_maxStencilNum = m_numVerts * maxCollisionPerVert;
	m_stencils.resize(m_maxStencilNum);
	m_stencilNum = 0;
}

void SeSchwarzPreconditioner::ComputeTotalAABB()
{
	m_aabb = SeAabb<SeVec3fSimd>();

	ComputeAABB(); // TODO: omp reduce
}


void SeSchwarzPreconditioner::ComputeAABB()
{

//MSVC compiler is stuck with OpenMP version 2.0, and unfortunately for you, reduction(max:) was only introduced with version 3.1 of the OpenMP C/C++ standard (that was in September 2011) 

//#pragma omp parallel for reduction(aabbAdd:m_aabb)
	for (int vid = 0; vid < m_numVerts; ++vid) // need omp custom reduction operation
	{
		m_aabb += m_positions[vid];
	}
}

// =========================================================
// SECTION 5.1: DOMAIN PARTITIONING (SPATIAL SORTING)
// =========================================================

// [對應論文 Section 5.1: Spatial Sorting]
// 論文提到："Similar to [Wu et al. 2015], we choose to sort the nodes by their Morton codes first."
// 目的：確保空間上相近的節點在記憶體中也是相鄰的，這是後續建立層級 (Hierarchy) 的基礎。
void SeSchwarzPreconditioner::SpaceSort()
{
	FillSortingData(); // 計算每個頂點的 Morton Code
	DoingSort();       // 根據 Code 進行排序
}

// [對應論文 Section 5.1]
// "We then obtain the 60-bit Morton code of each node by simply interleaving the bits of its cell indices."
void SeSchwarzPreconditioner::FillSortingData()
{
	OMP_PARALLEL_FOR

		// 計算 Morton Code 的實作

		for (int vid = 0; vid < m_numVerts; ++vid)
		{
			// [對應論文 Section 5.1]
            // 將座標正規化後編碼為 Morton Code
			SeVec3fSimd temp = (m_positions[vid] - m_aabb.Lower) / m_aabb.Extent();

			SeMorton64 mt;

			mt.Encode(temp.x, temp.y, temp.z);

			m_mortonCode[vid] = mt;

			m_MapperSortedGetOriginal[vid] = vid;
		}
}


void SeSchwarzPreconditioner::DoingSort()
{
	//==== sort m_MapperSortedGetOriginal by m_mortonCode1;

	std::sort(m_MapperSortedGetOriginal.begin(), m_MapperSortedGetOriginal.end(), [&](int i, int j) { return m_mortonCode[i] < m_mortonCode[j]; });
}

void SeSchwarzPreconditioner::ComputeInverseMapper()
{
	OMP_PARALLEL_FOR

		for (int vid = 0; vid < m_numVerts; ++vid)
		{
			int originalIndex = m_MapperSortedGetOriginal[vid];

			m_mapperOriginalGetSorted[originalIndex] = vid;
		}
}


void SeSchwarzPreconditioner::MapHessianTable()
{
	OMP_PARALLEL_FOR

		for (int vid = 0; vid < m_numVerts; ++vid)
		{
			const auto  originalIndex = m_MapperSortedGetOriginal[vid];

			m_mappedPos[vid] = m_positions[originalIndex];

			//====

			int numNeighbors = m_neighbours->Size(originalIndex);

			const int* neighbors = m_neighbours->IdxPtr(originalIndex);

			m_mappedNeighborsNum[vid] = numNeighbors + 1;	// include self

			m_mappedNeighbors[0][vid] = vid;				// include self

			for (int k = 1; k < numNeighbors + 1; ++k)
			{
				unsigned int originalNeighbors = neighbors[k - 1];//m_neighbors[k][originalIndex];

				m_mappedNeighbors[k][vid] = m_mapperOriginalGetSorted[originalNeighbors];
			}
		}
}

void SeSchwarzPreconditioner::MapCollisionStencilIndices()
{
	m_stencilIndexMapped.resize(m_stencilNum);

	OMP_PARALLEL_FOR

		for (int i = 0; i < m_stencilNum; i++)
		{
			const auto& s = m_stencils[i];

			for (int vi = 0; vi < s.verextNumPerStencil; vi++)
			{
				m_stencilIndexMapped[i][vi] = m_mapperOriginalGetSorted[s.index[vi]];
			}
		}
}

void SeSchwarzPreconditioner::PrepareCollisionStencils(const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts)
{
	int efNum = efCounts[m_numEdges];
	int eeNum = eeCounts[m_numEdges];
	int vfNum = vfCounts[m_numVerts];

	int totalStencilNum = efNum + eeNum + vfNum;

	if (totalStencilNum > m_maxStencilNum)
	{
		totalStencilNum = m_maxStencilNum;
		printf("stencil size %d exceed max stencils num  %d\n", totalStencilNum, m_maxStencilNum);
	}

	m_stencilNum = 0;

	OMP_PARALLEL_FOR

		for (int i = 0; i < totalStencilNum; i++)
		{
			Stencil s;

			if (i < efNum) //process ef
			{
				const auto& pair = efSets[i];

				if (pair.m_eId < 0 || pair.m_fId < 0) { continue; }

				const auto& edge = m_edges[pair.m_eId];
				const auto& face = m_faces[pair.m_fId];

				s.verextNumPerStencil = 5;
				s.vertexNumOfFirstPrimitive = 2;

				s.index[0] = edge[0];
				s.index[1] = edge[1];
				s.index[2] = face[0];
				s.index[3] = face[1];
				s.index[4] = face[2];

				s.weight[0] = pair.m_bary[0];
				s.weight[1] = 1.f - pair.m_bary[0];
				s.weight[2] = -pair.m_bary[1];
				s.weight[3] = -pair.m_bary[2];
				s.weight[4] = -(1.f - pair.m_bary[1] - pair.m_bary[2]);

				s.direction = pair.m_normal;

				s.stiff = pair.stiff;

			}
			else if (i < efNum + eeNum) // process ee
			{
				const auto& pair = eeSets[i];

				if (pair.m_eId1 < 0 || pair.m_eId0 < 0) { continue; }

				s.verextNumPerStencil = 4;
				s.vertexNumOfFirstPrimitive = 2;

				const auto& edge0 = m_edges[pair.m_eId0];
				const auto& edge1 = m_edges[pair.m_eId1];

				s.index[0] = edge0[0];
				s.index[1] = edge0[1];
				s.index[2] = edge1[0];
				s.index[3] = edge1[1];

				s.weight[0] = pair.m_bary[0];
				s.weight[1] = 1.f - pair.m_bary[0];
				s.weight[2] = -pair.m_bary[1];
				s.weight[3] = -(1.f - pair.m_bary[1]);

				s.direction = pair.m_normal;

				s.stiff = pair.stiff;
			}
			else if (i < totalStencilNum) //process vf
			{
				const auto& pair = vfSets[i];

				if (pair.m_vId < 0 || pair.m_fId < 0) { continue; }

				s.verextNumPerStencil = 4;
				s.vertexNumOfFirstPrimitive = 3;
				
				const auto& face = m_faces[pair.m_fId];
				
				s.index[0] = face[0];
				s.index[1] = face[1];
				s.index[2] = face[2];
				s.index[3] = pair.m_vId;

				s.weight[0] = -pair.m_bary[0];
				s.weight[1] = -pair.m_bary[1];
				s.weight[2] = -(1.f - pair.m_bary[2]);
				s.weight[3] = 1.f;

				s.direction = pair.m_normal;

				s.stiff = pair.stiff;
			}

			int stencilNum = Intrinsic::AtomicAdd(&m_stencilNum, 1);

			m_stencils[stencilNum] = s;
		}

	MapCollisionStencilIndices(); //1
}

// =========================================================
// SECTION 5.2: COARSE SPACE CONSTRUCTION
// =========================================================

// [對應論文 Section 5.2: Coarse Space Construction]
// 建構粗糙空間 (Coarse Space) 的層級。
// 對應論文描述的 "Aggregation-based coarsening"，將節點聚類成超節點 (Supernodes)。
void SeSchwarzPreconditioner::ReorderRealtime()
{
	Utility::MemsetZero(m_levelSize);

	BuildConnectMaskL0(); // 建立第 0 層的連接關係

	// [對應論文 Section 5.2 & Fig 6]
	// "This solution removes false coupling artifacts."
	// "check every domain-sized supernode... and split it... if they are not actually connected."
	// 這裡處理碰撞產生的額外連接，利用碰撞與拓撲連接資訊來修正 Morton Code 分群可能產生的錯誤耦合 (Artifacts)，避免不必要的耦合影響層級結構。
	BuildCollisionConnection(m_fineConnectMask.data(), nullptr); //2

	PreparePrefixSumL0();

	BuildLevel1(); // 建立第一層粗糙空間 C_(1)

	// [對應論文 Algorithm 2: Hierarchy Construction]
	// 迴圈對應論文中遞迴建立 l > 1 層級的過程。
	for (int level = 1; level < m_numLevel; level++)
	{
		Utility::MemsetZero(m_nextConnectMsk);

		BuildConnectMaskLx(level); // Coarsening connections

		// 在每一層都進行連通性檢查以避免錯誤聚合
		BuildCollisionConnection(m_nextConnectMsk.data(), m_CoarseSpaceTables[level - 1]);

		NextLevelCluster(level);

		PrefixSumLx(level);

		ComputeNextLevel(level);
	}

	TotalNodes();

	// [對應論文 Section 5.2]
	// "We collapse {map_l->l+1} to compute map_(l) from level 0 directly."
	// 建立直接從 Level 0 到任意 Level l 的映射表，方便並行計算。
	// 將多層映射表壓縮，讓程式可以直接查詢任一層的對應關係。
	AggregationKernel();
}

void SeSchwarzPreconditioner::BuildConnectMaskL0()
{
	const unsigned int warpDim = 32;

	int nWarps = (m_numVerts + warpDim - 1) / warpDim;

	OMP_PARALLEL_FOR

		for (int warpIdx = 0; warpIdx < nWarps; ++warpIdx)
		{
			for (int laneIdx = 0; laneIdx < warpDim; ++laneIdx)
			{
				int vId = warpIdx * warpDim + laneIdx;

				if (vId >= m_numVerts)
				{
					break;
				}

				int numNeighbors = m_mappedNeighborsNumRemain[vId];

				unsigned int connetMsk = 1U << laneIdx;	// include self

				int nk = 0;

				for (int k = 0; k < numNeighbors; k++)
				{
					int vIdConnected = m_mappedNeighborsRemain[k][vId];

					int warpIdxConnected = vIdConnected / warpDim;

					if (warpIdx == warpIdxConnected) // in the same warp
					{
						unsigned int laneIdxConnected = vIdConnected % warpDim;

						connetMsk |= (1U << laneIdxConnected);
					}
					else
					{
						m_mappedNeighborsRemain[nk][vId] = vIdConnected;
						nk++;
					}
				}

				m_mappedNeighborsNumRemain[vId] = nk;

				// distance cullling
				//const SeVec3fSimd& pos = m_mappedPos[vId];
				//for (int it = 0; it < Intrinsic::Popcount32(activeMsk); it++)
				//{
				//	const float th = AABB_TH * AABB_TH;
				//	CuFloat3 po(0.0f);
				//	po.x = Intrinsic::Shuffle(activeMsk, pos.x, it);
				//	po.y = Intrinsic::Shuffle(activeMsk, pos.y, it);
				//	po.z = Intrinsic::Shuffle(activeMsk, pos.z, it);
				//	if (Math::SqrLength(po - pos) < th)
				//	{
				//		connetMsk |= (1U << it);
				//	}
				//}

				m_fineConnectMask[vId] = connetMsk;
			}
		}
}


void SeSchwarzPreconditioner::BuildCollisionConnection(unsigned int* pConnect, const int* pCoarseTable)
{
	const int bank = 32;

	OMP_PARALLEL_FOR

		for (int i = 0; i < m_stencilNum; i++)
		{
			const auto& s = m_stencils[i];
			Int5 idx = m_stencilIndexMapped[i];
			unsigned int connMsk[5] = {};

			if (pCoarseTable)
			{
				for (int it = 0; it < s.verextNumPerStencil; it++)
				{
					idx[it] = pCoarseTable[idx[it]];
				}
			}

			for (int ita = 0; ita < s.verextNumPerStencil; ita++)
			{
				for (int itb = ita + 1; itb < s.verextNumPerStencil; itb++)
				{
					unsigned int myID = idx[ita];
					unsigned int otID = idx[itb];
					if (myID == otID) // redundant ?
					{
						continue;
					}
					if ((myID / bank == otID / bank))
					{
						if (ita < s.vertexNumOfFirstPrimitive && itb >= s.vertexNumOfFirstPrimitive)
						{
							connMsk[ita] |= (1U << (otID % bank));
							connMsk[itb] |= (1U << (myID % bank));
						}
					}
				}
			}

			for (int it = 0; it < s.verextNumPerStencil; it++)
			{
				if (connMsk[it])
				{
					Intrinsic::AtomicOr(&pConnect[idx[it]], connMsk[it]);
				}
			}
		}
}

void SeSchwarzPreconditioner::PreparePrefixSumL0()
{
	int warpDim = 32;

	int nWarps = (m_numVerts + warpDim - 1) / warpDim;

	OMP_PARALLEL_FOR

		for (int warpIdx = 0; warpIdx < nWarps; ++warpIdx)
		{
			unsigned int cacheMask[32];

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vId = laneIdx + warpIdx * warpDim;

				if (vId >= m_numVerts) { break; }

				cacheMask[laneIdx] = m_fineConnectMask[vId];
			}

			Intrinsic::SyncWarp();

			int prefixSum = 0;

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * warpDim;

				if (vid >= m_numVerts) { break; }

				unsigned int connetMsk = cacheMask[laneIdx];

				unsigned int visited = (1U << laneIdx);

				while (connetMsk != -1) // 
				{
					unsigned int todo = visited ^ connetMsk; // 

					if (!todo)
					{
						break;
					}

					unsigned int nextVisit = Intrinsic::Ffs(todo) - 1;

					visited |= (1U << nextVisit);

					connetMsk |= cacheMask[nextVisit];
				}

				m_fineConnectMask[vid] = connetMsk;

				unsigned int electedPrefix = Intrinsic::Popcount32(connetMsk & Intrinsic::LanemaskLt(laneIdx));

				if (electedPrefix == 0)
				{
					prefixSum++;
				}
			}

			m_prefixOrignal[warpIdx] = prefixSum;
		}
}

void SeSchwarzPreconditioner::BuildLevel1()
{
	constexpr unsigned int blockDim = 1024;//32*32

	int nBlocks = (m_numVerts + blockDim - 1) / blockDim;

	OMP_PARALLEL_FOR

		for (int blockIdx = 0; blockIdx < nBlocks; ++blockIdx)
		{
			unsigned int warpPrefix[32];

			unsigned int theBlockGlobalPrefix = 0;

			unsigned int globalWarpOffset = blockIdx * 32;

			for (int warpIdx = 0; warpIdx < blockIdx; ++warpIdx)
			{
				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					theBlockGlobalPrefix += m_prefixOrignal[threadIdx];
				}
			}

			for (int warpIdx = 0; warpIdx < 32; ++warpIdx)
			{
				warpPrefix[warpIdx] = m_prefixOrignal[globalWarpOffset + warpIdx];
			}

			Intrinsic::SyncBlock();

			for (unsigned int warpIdx = 0; warpIdx < 32; ++warpIdx)
			{
				unsigned int theWarpGlobalPrefix = theBlockGlobalPrefix;

				for (unsigned int prevWarpIdx = 0; prevWarpIdx < warpIdx; ++prevWarpIdx)
				{
					theWarpGlobalPrefix += warpPrefix[prevWarpIdx];
				}


				unsigned int electedMask = 0;

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					int vid = threadIdx + blockIdx * blockDim;

					if (vid >= m_numVerts) { break; }


					if (vid == m_numVerts - 1)
					{
						m_levelSize[1].x = theWarpGlobalPrefix + warpPrefix[warpIdx];
						m_levelSize[1].y = (m_numVerts + 31) / 32 * 32;
					}

					unsigned int connMsk = m_fineConnectMask[vid];

					unsigned int electedPrefix = Intrinsic::Popcount32(connMsk & Intrinsic::LanemaskLt(laneIdx));

					if (electedPrefix == 0)
					{
						electedMask |= 1U << laneIdx;
					}
					//unsigned int electedMask = Intrinsic::BallotSync(-1, electedPrefix == 0);
				}


				unsigned int lanePrefix[32];

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					int vid = threadIdx + blockIdx * blockDim;

					if (vid >= m_numVerts) { break; }


					lanePrefix[laneIdx] = Intrinsic::Popcount32(electedMask & Intrinsic::LanemaskLt(laneIdx));

					lanePrefix[laneIdx] += theWarpGlobalPrefix;
				}


				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					int vid = threadIdx + blockIdx * blockDim;

					if (vid >= m_numVerts) { break; }


					unsigned int connMsk = m_fineConnectMask[vid];

					unsigned int elected_lane = Intrinsic::Ffs(connMsk) - 1;

					unsigned int theLanePrefix = lanePrefix[elected_lane];

					m_CoarseSpaceTables[0][vid] = theLanePrefix;

					m_goingNext[vid] = theLanePrefix + (m_numVerts + 31) / 32 * 32;
				}
			}
		}
}


void SeSchwarzPreconditioner::BuildConnectMaskLx(int level)
{
	constexpr unsigned int warpDim = 32;

	int nWarps = (m_numVerts + warpDim - 1) / warpDim;

	const unsigned int bank = 32;

	OMP_PARALLEL_FOR

		for (int warpIdx = 0; warpIdx < nWarps; ++warpIdx)
		{
			unsigned int prefixMsk[32];
			unsigned int connectMsk[32];

			for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= m_numVerts) { break; }

				prefixMsk[laneIdx] = m_fineConnectMask[vid];

				unsigned int coarseVid = m_CoarseSpaceTables[level - 1][vid];

				connectMsk[laneIdx] = 0;// 1U << (coarseVid % bank);	//==== include self

				unsigned int kn = m_mappedNeighborsNumRemain[vid];

				unsigned int nk = 0;

				for (unsigned int k = 0; k < kn; k++)
				{
					unsigned int connect = m_mappedNeighborsRemain[k][vid];

					unsigned int coarseConnect = m_CoarseSpaceTables[level - 1][connect];

					if (coarseVid / bank == coarseConnect / bank)
					{
						unsigned int off = coarseConnect % bank;

						connectMsk[laneIdx] |= (1U << off);
					}
					else
					{
						m_mappedNeighborsRemain[nk][vid] = connect;
						nk++;
					}
				}

				m_mappedNeighborsNumRemain[vid] = nk;
			}

			bool isFullWarp = (prefixMsk[0] == -1);

			if (isFullWarp)
			{
				unsigned int connectMskFull = 0;

				for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					connectMskFull |= connectMsk[laneIdx];
				}
				for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					connectMsk[laneIdx] = connectMskFull;
				}
			}
			else
			{
				unsigned int cacheMsk[32];

				for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					cacheMsk[laneIdx] = 0;
				}

				unsigned int electedLane[32];

				for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					electedLane[laneIdx] = Intrinsic::Ffs(prefixMsk[laneIdx]) - 1;

					if (connectMsk[laneIdx])
					{
						//cacheMsk[electedLane[laneIdx]] |= connectMsk[laneIdx];
						Intrinsic::AtomicOr(&cacheMsk[electedLane[laneIdx]], connectMsk[laneIdx]);
					}
				}

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					connectMsk[laneIdx] = cacheMsk[electedLane[laneIdx]];
				}
			}

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= m_numVerts) { break; }

				unsigned int coarseVid = m_CoarseSpaceTables[level - 1][vid];

				unsigned int electedPrefix = Intrinsic::Popcount32(prefixMsk[laneIdx] & Intrinsic::LanemaskLt(laneIdx));

				if (connectMsk[laneIdx] && electedPrefix == 0)
				{
					Intrinsic::AtomicOr(m_nextConnectMsk.data() + coarseVid, connectMsk[laneIdx]);
				}
			}
		}
}

void SeSchwarzPreconditioner::NextLevelCluster(int level)
{
	const int levelNum = m_levelSize[level].x;

	const int warpDim = 32;

	int numWarps = (m_numVerts + warpDim - 1) / warpDim;

	int maxWarpIdx = (levelNum + warpDim - 1) / warpDim;

	OMP_PARALLEL_FOR

		for (int warpIdx = 0; warpIdx < numWarps; ++warpIdx)
		{
			unsigned int cachedMsk[32];

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= (levelNum + 31) / 32 * 32)
				{
					break;// early culling
				}

				unsigned int connectMsk = (1U << laneIdx);

				if (vid < levelNum)
				{
					connectMsk |= m_nextConnectMsk[vid];// connect to current 
				}

				cachedMsk[laneIdx] = connectMsk;

				if (vid >= levelNum)
				{
					break;
				}
			}

			Intrinsic::SyncWarp();

			unsigned int prefixSum = 0;

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= levelNum || vid >= (levelNum + 31) / 32 * 32)
				{
					break;// early culling
				}

				unsigned int connectMsk = cachedMsk[laneIdx];

				unsigned int visited = (1U << laneIdx);

				while (true)
				{
					unsigned int todo = visited ^ connectMsk;

					if (!todo)
					{
						break;
					}

					unsigned int nextVisist = Intrinsic::Ffs(todo) - 1;

					visited |= (1U << nextVisist);

					connectMsk |= cachedMsk[nextVisist];
				}

				m_nextConnectMsk[vid] = connectMsk;

				unsigned int electedPrefix = Intrinsic::Popcount32(connectMsk & Intrinsic::LanemaskLt(laneIdx));

				if (electedPrefix == 0)
				{
					prefixSum++;
				}
			}

			if (warpIdx < maxWarpIdx)
			{
				m_nextPrefix[warpIdx] = prefixSum;
			}
		}
}

void SeSchwarzPreconditioner::PrefixSumLx(int level)
{
	const int levelNum = m_levelSize[level].x;

	const int levelBegin = m_levelSize[level].y;

	unsigned int blockDim = 1024;

	int nBlocks = (m_numVerts + blockDim - 1) / blockDim;

	OMP_PARALLEL_FOR

		for (int blockIdx = 0; blockIdx < nBlocks; ++blockIdx)
		{
			unsigned int warpPrefix[32];

			unsigned int theBlockPrefix = 0;

			unsigned int globalWarpOffset = blockIdx * 32;

			for (int warpIdx = 0; warpIdx < blockIdx; ++warpIdx)
			{
				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					unsigned int vid = threadIdx + blockIdx * blockDim;

					if (vid >= (levelNum + blockDim - 1) / blockDim * blockDim)
					{
						break;
					}

					theBlockPrefix += m_nextPrefix[threadIdx];
				}
			}

			for (int warpIdx = 0; warpIdx < 32; ++warpIdx)
			{
				warpPrefix[warpIdx] = m_nextPrefix[globalWarpOffset + warpIdx];
			}

			Intrinsic::SyncBlock();

			for (unsigned int warpIdx = 0; warpIdx < 32; ++warpIdx)
			{
				unsigned int theWarpPrefix = theBlockPrefix;

				for (unsigned int prevWarpIdx = 0; prevWarpIdx < warpIdx; ++prevWarpIdx)
				{
					theWarpPrefix += warpPrefix[prevWarpIdx];
				}

				unsigned int electedMsk = 0;

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					int vid = threadIdx + blockIdx * blockDim;

					if (vid >= levelNum)
					{
						break;
					}
					if (vid == levelNum - 1)
					{
						m_levelSize[level + 1].x = theWarpPrefix + warpPrefix[warpIdx];
						m_levelSize[level + 1].y = levelBegin + (levelNum + 31) / 32 * 32;
					}

					unsigned int connMsk = m_nextConnectMsk[vid];

					unsigned int electedPrefix = Intrinsic::Popcount32(connMsk & Intrinsic::LanemaskLt(laneIdx));

					if (electedPrefix == 0)
					{
						electedMsk |= (1U << laneIdx);
					}

					//unsigned int electedMask = Intrinsic::BallotSync(-1, electedPrefix == 0);
				}

				unsigned int lanePrefix[32];

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					lanePrefix[laneIdx] = Intrinsic::Popcount32(electedMsk & Intrinsic::LanemaskLt(laneIdx));
					lanePrefix[laneIdx] += theWarpPrefix;
				}

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					unsigned int threadIdx = laneIdx + warpIdx * 32;

					int vid = threadIdx + blockIdx * blockDim;

					if (vid >= levelNum) { break; }

					int electedLane = Intrinsic::Ffs(m_nextConnectMsk[vid]) - 1;

					unsigned int theLanePrefix = lanePrefix[electedLane];

					m_nextConnectMsk[vid] = theLanePrefix;

					m_goingNext[vid + levelBegin] = theLanePrefix + levelBegin + (levelNum + 31) / 32 * 32;
				}
			}
		}
}

void SeSchwarzPreconditioner::ComputeNextLevel(int level)
{
	OMP_PARALLEL_FOR

		for (int vid = 0; vid < m_numVerts; ++vid)
		{
			int next = m_CoarseSpaceTables[level - 1][vid];

			m_CoarseSpaceTables[level][vid] = m_nextConnectMsk[next];
		}
}

void SeSchwarzPreconditioner::TotalNodes()
{
	m_totalNumberClusters = m_levelSize[m_numLevel].y;
	//printf("%d\n", m_totalNumberClusters);
}

void SeSchwarzPreconditioner::AggregationKernel()
{
	const int warpDim = 32;

	int numWarps = (m_numVerts + warpDim - 1) / warpDim;

	OMP_PARALLEL_FOR

		for (int warpIdx = 0; warpIdx < numWarps; ++warpIdx)
		{
			unsigned int firstInWarp = warpIdx * 32;

			int curr[32];
			int next[32];
			int aggLevel[32];

			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= m_numVerts) { break; }

				curr[laneIdx] = vid;

				aggLevel[laneIdx] = m_numLevel - 1;
			}

			Int4 coarseTable[32];

			for (int l = 0; l < m_numLevel - 1; ++l)
			{
				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					next[laneIdx] = m_goingNext[curr[laneIdx]];
				}

				Intrinsic::SyncWarp();

				for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
				{
					int vid = laneIdx + warpIdx * 32;

					if (vid >= m_numVerts) { break; }

					if (next[laneIdx] == next[0])
					{
						aggLevel[laneIdx] = Math::Min(l, aggLevel[laneIdx]);
					}

					curr[laneIdx] = next[laneIdx];
					coarseTable[laneIdx][l] = next[laneIdx];
				}
			}


			for (unsigned int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + warpIdx * 32;

				if (vid >= m_numVerts) { break; }

				m_denseLevel[vid] = aggLevel[laneIdx];

				m_coarseTables[vid] = coarseTable[laneIdx];
			}
		}
}

void SeSchwarzPreconditioner::AdditionalSchwarzHessian2(SeMatrix3f hessian, std::vector<SeMatrix3f>& pAdditionalHessian, SeArray2D<SeMatrix3f>& pDenseHessian, int v1, int v2, const std::vector<int>& pGoingNext, int nLevel, int vNum)
{
	int level = 0;
	unsigned int myID = v1;
	unsigned int otID = v2;
	const unsigned int bank = 32;

	while (myID / bank != otID / bank && level < nLevel)
	{
		myID = pGoingNext[myID];
		otID = pGoingNext[otID];
		level++;
	}

	if (level >= nLevel)
		return;

	Intrinsic::AtomicAdd(&pDenseHessian[otID % bank][myID], hessian);
	Intrinsic::AtomicAdd(&pDenseHessian[myID % bank][otID], hessian);

	if (level < nLevel - 1)
	{
		myID = pGoingNext[myID];
		otID = pGoingNext[otID];

		if (myID == otID)
		{
			Intrinsic::AtomicAdd(&pAdditionalHessian[myID], hessian * 2.0f);
		}
		else
		{
			Intrinsic::AtomicAdd(&pAdditionalHessian[myID], hessian);
			Intrinsic::AtomicAdd(&pAdditionalHessian[otID], hessian);
		}
	}
}

void SeSchwarzPreconditioner::PrepareCollisionHessian()
{
	OMP_PARALLEL_FOR

		for (int i = 0; i < m_stencilNum; i++)
		{
			const auto& s = m_stencils[i];
			auto idx = m_stencilIndexMapped[i];

			Float3 d(s.direction[0], s.direction[1], s.direction[2]);

			SeMatrix3f hessian = OuterProduct(d, d * s.stiff);

			for (int it = 0; it < s.verextNumPerStencil; it++)
			{
				Intrinsic::AtomicAdd(&m_additionalHessian32[idx[it]], hessian * Math::Square(s.weight[it]));
			}

			for (int ita = 0; ita < s.verextNumPerStencil; ita++)
			{
				for (int itb = ita + 1; itb < s.verextNumPerStencil; itb++)
				{
					AdditionalSchwarzHessian2(s.weight[ita] * s.weight[itb] * hessian, m_additionalHessian32, m_hessian32, idx[ita], idx[itb], m_goingNext, m_numLevel, m_numVerts);
				}
			}
		}
}

// =========================================================
// SECTION 6: MATRIX PRECOMPUTATION
// =========================================================

// [對應論文 Algorithm 1: Matrix Precomputation]
// 負責組裝子區域矩陣 A_{d,(l)} = S_d * A_{(l)} * S_d^T

void SeSchwarzPreconditioner::PrepareHessian(const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges)
{

	// [對應論文 Section 6.1] "In this work, we set M = 32"
	const unsigned int bank = 32; 

	int nVC = (m_numVerts + 31) / 32 * 32;

	OMP_PARALLEL_FOR

		// [對應論文 Section 6.1: Matrix Assembly]
		// "We check every 3x3 system matrix block A[i,j]... and add it into the... sub-matrix A_{d,(l)}"
		// 程式碼遍歷矩陣區塊，根據層級映射 (m_goingNext) 將數值累加到對應層級的 Hessian 中。
		for (int vid = nVC; vid < m_totalNumberClusters; ++vid)
		{
			auto oldDiagonal = m_additionalHessian32[vid];
			int myID = vid;
			Intrinsic::AtomicAdd(&m_hessian32[myID % bank][myID], oldDiagonal);
			while (true)
			{
				myID = m_goingNext[myID];
				if (myID >= m_totalNumberClusters)
				{
					break;
				}
				Intrinsic::AtomicAdd(&m_hessian32[myID % bank][myID], oldDiagonal);
			}
		}

	int nblock = (m_numVerts + 31) / 32;
	OMP_PARALLEL
	{
		std::vector<std::unordered_map<int,SeMatrix3f>> diagTable(m_numLevel);
		
#pragma omp for
		for (int bid = 0; bid < nblock; ++bid)
		{
			for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + bid * 32;
				if (vid >= m_numVerts)
					break;
				const int vidOrig = m_MapperSortedGetOriginal[vid];
				int oldNum = m_mappedNeighborsNum[vid];

				auto oldDiagonal = diagonal[vidOrig] + m_additionalHessian32[vid];
				m_hessian32[vid % bank][vid] += oldDiagonal;   // self diagonal

				// [對應論文 Algorithm 1]
                // "check every 3x3 system matrix block A[i,j] and add it into the... sub-matrix A_{d,(l)}" 
                // 這裡遍歷 CSR 格式的稀疏矩陣，將數值累加到對應的 m_hessian32 (即 A_{d,(l)})
				for (int k = 1; k < oldNum; k++)
				{
					const unsigned int neighbor = m_mappedNeighbors[k][vid];
					SeMatrix3f mat = csrOffDiagonals[csrRanges[vidOrig] + k - 1];
					int level = 0;
					unsigned int levelSz = m_numVerts;
					unsigned int myID = vid;
					unsigned int otID = neighbor;

					// 根據層級映射表 m_goingNext 找到對應的粗糙層級與 Domain ID
					while (myID / bank != otID / bank && level < m_numLevel)
					{
						level++;
						myID = m_goingNext[myID];
						otID = m_goingNext[otID];
					}
					if (level >= m_numLevel)
					{
						continue;
					}

					// 累加矩陣數值 (AtomicAdd 用於平行處理衝突)
					if (level <= 1) // 按照block分并行，level 1的位置也一定属于本线程
						m_hessian32[otID % bank][myID] += mat;
					else
						Intrinsic::AtomicAdd(&m_hessian32[otID % bank][myID], mat);

					if (level == 0)
						oldDiagonal += mat;
					else if (level + 1 < m_numLevel)
					{
						myID = m_goingNext[myID];
						auto& table = diagTable[level + 1];
						if (table.find(myID) == table.end())
							table[myID] = mat;
						else
							table[myID] += mat;
					}
				}
				if (1 < m_numLevel)
				{
					int myID = m_goingNext[vid];
					m_hessian32[myID % bank][myID] += oldDiagonal;   // self diagonal
					if (2 < m_numLevel)
					{
						myID = m_goingNext[myID];
						auto& table = diagTable[2];
						if (table.find(myID) == table.end())
							table[myID] = oldDiagonal;
						else
							table[myID] += oldDiagonal;
					}
				}
			}
		}

		for (int lv = 2; lv < m_numLevel; lv++)
		{
			const auto& table = diagTable[lv];
			for (auto& pair : table)
			{
				int myID = pair.first;
				Intrinsic::AtomicAdd(&m_hessian32[myID % bank][myID],pair.second);
				if (lv + 1 < m_numLevel)
				{
					myID = m_goingNext[myID];
					auto& nexttable = diagTable[lv + 1];
					if (nexttable.find(myID) == nexttable.end())
						nexttable[myID] = pair.second;
					else
						nexttable[myID] += pair.second;
				}
			}
		}
	}
}

// [對應論文 Section 6.2: Fast Sub-Matrix Inversion]
// Per-Domain Sub-Matrix Inverse Calculation
// [對應論文 Algorithm 1 (Lines 10-12)]
// 計算子矩陣的逆：L^{-T} D^{-1} L^{-1}
void SeSchwarzPreconditioner::LDLtInverse512()
{
	const int triSz = (1 + 96) * 96 / 2 + 16 * 3;

	const int activeblockNum = m_totalNumberClusters / 32;

	OMP_PARALLEL_FOR

		for (int block = 0; block < activeblockNum; block++)
		{
			float A[96][96] = {}; // 1. 將資料載入暫存器 (A[][]), 載入 96x96 的子矩陣 A_{d,(l)} (因為 M=32, 32*3=96)

			for (int x = 0; x < 32; x++)
			{
				for (int y = 0; y < 32; y++)
				{
					SeMatrix3f temp = m_hessian32[y][x + block * 32];

					if (x == y && temp(0, 0) == 0.0f)
					{
						temp = SeMatrix3f::Identity();
					}
					for (int ii = 0; ii < 3; ii++)
					{
						for (int jj = 0; jj < 3; jj++)
						{
							A[x * 3 + ii][y * 3 + jj] = temp(ii, jj);
						}
					}
				}
			}

			//const int pt = -1;//212;
			//if (block == pt)
			//{
			//	std::ofstream ofs("c:\\test\\MRF.txt");
			//	for (int y = 0; y < 96; y++)
			//	{
			//		for (int x = 0; x < 96; x++)
			//		{
			//			ofs << A[y][x] << " ";
			//		}
			//		ofs << std::endl;
			//	}
			//}

			// 向下消元
#if defined(WIN32) || (defined(__AVX2__) && defined(__FMA__))
			// 步驟 1: 高斯消去法 (Gauss-Jordan Elimination)
            // [對應論文 Section 6.2] "We apply Gauss-Jordan elimination... to get [U | L^-1]"
				
			// 使用 AVX 指令集進行 SIMD 平行化運算優化消去過程
			for (int x = 0; x < 96; x++)
			{
				float diag = A[x][x];
				__m256 line[12];
				for (int it = 0; it < 12; it++)
					line[it] = _mm256_loadu_ps(A[x] + it * 8);

				for (int y = x + 1; y < 96; y++)
				{
					if (A[y][x] == 0.0f)
						continue;
					float r = -A[y][x] / diag;
					__m256 ratio = _mm256_set1_ps(r);
					for (int it = 0; it < 12; it++)
					{
						__m256 temp = _mm256_fmadd_ps(ratio, line[it], _mm256_loadu_ps(A[y] + it * 8)); // A[y] += A[x] * ratio 
						_mm256_storeu_ps(A[y] + it * 8, temp);
					}
					A[y][x] = r;
				}
			}


			// inv diagonal 
			float diagonal[96] = {};
			for (int y = 0; y < 96; y++)
			{
				diagonal[95 - y] = A[y][y];
				A[y][y] = 1.0f;
				for (int x = y + 1; x < Math::Min(96, y + 9); x++)
				{
					A[y][x] = 0.0f;
				}
			}
			for (int it = 0; it < 12; it++)
			{
				__m256 temp = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_loadu_ps(diagonal + it * 8));
				_mm256_storeu_ps(diagonal + it * 8, temp);
			}

			int off = block * triSz;
			// output diagonal 
			// 步驟 2: 計算 L^{-T} D^{-1} L^{-1}
			// [對應論文 Section 6.2] "In the third step, we calculate L^{-T} D^{-1} L^{-1} by visiting every two columns..."
			// 利用 SIMD 計算對稱矩陣乘積
			// 這裡計算出的結果存入 m_invSymR，使用論文 Fig 11 描述的 Compact Storage Format。
			for (int it = 0; it < 12; it++)
			{
				int lc = 96 - it * 8;
				__m256 acc = _mm256_setzero_ps();
				for (int l = 0; l < lc; l++)
				{
					__m256 a = _mm256_loadu_ps(A[95 - l] + it * 8); 
					a = _mm256_mul_ps(a, a);
					__m256 r = _mm256_set1_ps(diagonal[l]);
					acc = _mm256_fmadd_ps(r, a, acc);
				}
				_mm256_storeu_ps(&m_invSymR[off + it * 8], acc); // 存入 Compact Format
			}
			off += 12 * 8;
			for (int it = 0; it < 12; it++)
			{
				int xBg = it * 8;
				for (int scan = xBg + 1; scan <= (96 - 8); scan++)
				{
					int lc = 96 - scan;
					__m256 acc = _mm256_setzero_ps();
					for (int l = 0; l < lc; l++)
					{
						__m256 a = _mm256_loadu_ps(A[95 - l] + xBg);
						__m256 b = _mm256_loadu_ps(A[95 - l] + scan);
						a = _mm256_mul_ps(a, b);
						__m256 r = _mm256_set1_ps(diagonal[l]);
						acc = _mm256_fmadd_ps(r, a, acc);
					}
					_mm256_storeu_ps(&m_invSymR[off], acc);
					off += 8;
				}
			}
#else
			// Scalar fallback (equivalent to the AVX path above).

			// Step 1: LDLt factorization (in-place Gauss elimination).
			// After this loop, A's strict lower triangle holds L factors (negated ratios),
			// and the upper triangle + diagonal hold U.
			for (int x = 0; x < 96; x++)
			{
				float diag = A[x][x];
				for (int y = x + 1; y < 96; y++)
				{
					if (A[y][x] == 0.0f)
						continue;
					float r = -A[y][x] / diag;
					for (int col = 0; col < 96; col++)
						A[y][col] += r * A[x][col];
					A[y][x] = r;
				}
			}

			// Extract diagonal of U (reversed), set upper triangle entries to 0/1
			// so that what remains in each row is L^{-1}.
			float diagonal[96] = {};
			for (int y = 0; y < 96; y++)
			{
				diagonal[95 - y] = A[y][y];
				A[y][y] = 1.0f;
				for (int x = y + 1; x < 96 && x < y + 9; x++)
					A[y][x] = 0.0f;
			}
			// Invert diagonal: diagonal[i] = 1.0 / diagonal[i]
			for (int i = 0; i < 96; i++)
				diagonal[i] = 1.0f / diagonal[i];

			int off = block * triSz;

			// Step 2: Compute L^{-T} D^{-1} L^{-1} and store in compact format.

			// (a) Diagonal entries of the inverse (8 values at a time, 12 groups).
			for (int it = 0; it < 12; it++)
			{
				int lc = 96 - it * 8;
				for (int lane = 0; lane < 8; lane++)
				{
					float acc = 0.0f;
					int col = it * 8 + lane;
					for (int l = 0; l < lc; l++)
					{
						float a = A[95 - l][col];
						acc += diagonal[l] * a * a;
					}
					m_invSymR[off + it * 8 + lane] = acc;
				}
			}
			off += 12 * 8;

			// (b) Off-diagonal "body" entries (groups of 8).
			for (int it = 0; it < 12; it++)
			{
				int xBg = it * 8;
				for (int scan = xBg + 1; scan <= (96 - 8); scan++)
				{
					int lc = 96 - scan;
					for (int lane = 0; lane < 8; lane++)
					{
						float acc = 0.0f;
						for (int l = 0; l < lc; l++)
						{
							float a = A[95 - l][xBg + lane];
							float b = A[95 - l][scan + lane];
							acc += diagonal[l] * a * b;
						}
						m_invSymR[off + lane] = acc;
					}
					off += 8;
				}
			}
#endif
			for (int it = 0; it < 12; it++)
			{
				for (int lane = 0; lane < 7; lane++)
				{
					int xBg = it * 8 + lane;
					for (int h = (96 - 7 + lane); h < 96; h++)
					{
						int lc = 96 - h;
						float acc = 0.f;
						for (int l = 0; l < lc; l++)
						{
							float a = A[95 - l][xBg];
							float b = A[95 - l][h];
							float r = diagonal[l];
							acc += a * b * r;
						}
						m_invSymR[off] = acc;
						off++;
					}
				}
			}

			//if (block == pt)
			//{
			//	std::ofstream ofs("c:\\test\\MRF3.txt");
			//	float B[96][96] = {};
			//	int Noff = block * triSz;
			//	for (int i = 0; i < 96; i++)
			//	{
			//		B[i][i] = m_invSymR[i + Noff];
			//	}
			//	Noff += 96;
			//  // body
			//	for (int it = 0; it < 12; it++)
			//	{
			//		int xBg = it * 8;
			//		for (int scan = xBg + 1; scan <= (96 - 8); scan++)
			//		{
			//			for (int l = 0; l < 8; l++)
			//			{
			//				float value = m_invSymR[Noff + l];
			//				B[scan + l][xBg + l] = value;
			//			}
			//			Noff += 8;
			//		}
			//	}
			//  // remainder
			//	for (int it = 0; it < 12; it++)
			//	{
			//		for (int lane = 0; lane < 7; lane++)
			//		{
			//			int xBg = it * 8 + lane;
			//			for (int h = (96 - 7 + lane); h < 96; h++)
			//			{
			//				float value = m_invSymR[Noff];
			//				Noff++;
			//				B[h][xBg] = value;
			//			}
			//		}
			//	}
			//
			//	for (int y = 0; y < 96; y++)
			//	{
			//		for (int x = 0; x < 96; x++)
			//		{
			//			ofs << B[y][x] << " ";
			//		}
			//		ofs << std::endl;
			//	}
			//}
		}
}

void SeSchwarzPreconditioner::BuildResidualHierarchy(const SeVec3fSimd* m_cgResidual)
{
	int nblock = (m_numVerts + 31) / 32;

	OMP_PARALLEL
	{
		std::vector<std::unordered_map<int,SeVec3fSimd>> diagTable(m_numLevel);
		
#pragma omp for

		for (int bid = 0; bid < nblock; ++bid)
		{
			for (int lane = 0; lane < 32; lane++)
			{
				int vid = lane + bid * 32;
				if (vid >= m_numVerts)
					break;
				unsigned int vidOrig = m_MapperSortedGetOriginal[vid];
				SeVec3fSimd r = m_cgResidual[vidOrig];
				m_mappedR[vid] = r;
				if (1 < m_numLevel)
				{
					int next = m_goingNext[vid];
					m_mappedR[next] += r;
				}
			}
		}
	}

		if (m_numLevel > 2)
		{
			int bg = m_levelSize[1].y;
			int ed = m_levelSize[2].y;
			for (int vid = bg; vid < ed; vid++)
			{
				SeVec3fSimd r = m_mappedR[vid];
				int nx = vid;
				for (int lv = 2; lv < m_numLevel; lv++)
				{
					nx = m_goingNext[nx];
					m_mappedR[nx] += r;
				}
			}
		}

	//printf("level %d\n", m_nLevel);

	//for (int l = 0; l < m_nLevel + 1; l++)
	//	printf("%d ",m_levelSize[l].y);
		//printf("\n");
}

// =========================================================
// SECTION 7.1: PER-DOMAIN PRECONDITIONING
// =========================================================

// [對應論文 Section 7.1]
// [對應論文 Fig 12 & 13: Symmetric Matrix-Vector Multiplication]
// "Our solution is a novel symmetric-matrix-vector multiplication method with balanced workload and zero write conflict."
void SeSchwarzPreconditioner::SchwarzLocalXSym()
{
	constexpr int blockDim = 32;

	const int activeblockNum = m_totalNumberClusters / blockDim;

	const int triSz = (1 + 96) * 96 / 2 + 16 * 3;

	OMP_PARALLEL_FOR

		for (int blockIdx = 0; blockIdx < activeblockNum; ++blockIdx)
		{
			float cacheRhs[96];
			float cacheOut[96];
			for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				// 載入 RHS
				int vid = laneIdx + blockIdx * blockDim;
				auto rhs = m_mappedR[vid];
				cacheRhs[laneIdx * 3 + 0] = rhs.x;
				cacheRhs[laneIdx * 3 + 1] = rhs.y;
				cacheRhs[laneIdx * 3 + 2] = rhs.z;
			}
#if defined(WIN32) || (defined(__AVX2__) && defined(__FMA__))
			// ------------------------------------
			// diagnal
			// Phase 1: 對角線處理 (Diagonal processing)
			// [對應論文 Fig 12 左圖] "process 32 matrix rows backward from the diagonal"
			int off = blockIdx * triSz;
			for (int it = 0; it < 12; it++)
			{
				__m256 diag = _mm256_loadu_ps(&m_invSymR[off + it * 8]);
				__m256 diagRhs = _mm256_loadu_ps(&cacheRhs[it * 8]);
				diag = _mm256_mul_ps(diag, diagRhs);
				_mm256_storeu_ps(&cacheOut[it * 8], diag);
			}

			// Phase 2: 非對角線處理 (Off-diagonal upward processing)
            // [對應論文 Fig 12 右圖] "process matrix columns upward..."
			off += 8 * 12;  // 這行結束後，開始讀取 off-diagonal 區(上三角)
			for (int it = 0; it < 11; it++)
			{
				int xBg = it * 8;
				__m256 sigularRhs = _mm256_loadu_ps(&cacheRhs[xBg]);
				__m256 sigularResult = _mm256_setzero_ps();
				for (int scan = xBg + 1; scan <= (96 - 8); scan++)
				{
					__m256 mtx = _mm256_loadu_ps(&m_invSymR[off]); // 讀 off-diagonal
					off += 8;

					__m256 scanRhs = _mm256_loadu_ps(&cacheRhs[scan]);
					sigularResult = _mm256_fmadd_ps(mtx, scanRhs, sigularResult);

					// 利用對稱性同時更新兩邊
					__m256 scanResult = _mm256_loadu_ps(&cacheOut[scan]);
					scanResult = _mm256_fmadd_ps(mtx, sigularRhs, scanResult); // 使用 AVX 指令 _mm256_fmadd_ps 進行乘加運算, 對應「upward處理欄」的更新
					_mm256_storeu_ps(&cacheOut[scan], scanResult);
				}

				__m256 sigularResultOrg = _mm256_loadu_ps(&cacheOut[xBg]);
				sigularResult = _mm256_add_ps(sigularResultOrg, sigularResult);
				_mm256_storeu_ps(&cacheOut[xBg], sigularResult);
			}
#else
			// Scalar fallback (equivalent to the AVX path above).

			// Phase 1: Diagonal — cacheOut[i] = invSymR_diag[i] * cacheRhs[i]
			int off = blockIdx * triSz;
			for (int i = 0; i < 96; i++)
				cacheOut[i] = m_invSymR[off + i] * cacheRhs[i];

			// Phase 2: Off-diagonal body (groups of 8, symmetric update)
			off += 8 * 12;
			for (int it = 0; it < 11; it++)
			{
				int xBg = it * 8;
				for (int scan = xBg + 1; scan <= (96 - 8); scan++)
				{
					for (int lane = 0; lane < 8; lane++)
					{
						float mtx = m_invSymR[off + lane];
						cacheOut[xBg + lane]  += mtx * cacheRhs[scan + lane];
						cacheOut[scan + lane] += mtx * cacheRhs[xBg + lane];
					}
					off += 8;
				}
			}
#endif

			for (int it = 0; it < 12; it++)
			{
				for (int lane = 0; lane < 7; lane++)
				{
					int xBg = it * 8 + lane;
					float sigularRhs = cacheRhs[xBg];
					float singularResult = 0.0f;
					for (int h = (96 - 7 + lane); h < 96; h++)
					{
						float value = m_invSymR[off]; // 讀上三角尾端
						off++;
						if (value)
						{
							singularResult += value * cacheRhs[h];
							cacheOut[h] += value * sigularRhs;  // 同樣是 off-diagonal 更新
						}
					}
					cacheOut[xBg] += singularResult;
				}
			}


			for (int laneIdx = 0; laneIdx < 32; ++laneIdx)
			{
				int vid = laneIdx + blockIdx * blockDim;

				SeVec3fSimd out(0.0f);
				out.x = cacheOut[laneIdx * 3 + 0];
				out.y = cacheOut[laneIdx * 3 + 1];
				out.z = cacheOut[laneIdx * 3 + 2];

				m_mappedZ[vid] = out; // 將結果存回 m_mappedZ (對應各層的解向量 y_(l))
			}

		}
}

// [對應論文 Equation 5 的加總]
// [對應論文 Section 7] "Summing up the results at all levels into a joint output."
void SeSchwarzPreconditioner::CollectFinalZ(SeVec3fSimd* m_cgZ)
{
	OMP_PARALLEL_FOR

		for (int vid = 0; vid < m_numVerts; ++vid)
		{
			unsigned int mappedIndex = m_MapperSortedGetOriginal[vid];

			auto z = m_mappedZ[vid]; // 取出第 0 層結果

			Int4 table = m_coarseTables[vid];

			// 累加粗糙層級 (Coarser Levels, Levels 1 to L) 的修正量
			// 對應公式中的 Sum
			for (int l = 1; l < Math::Min(m_numLevel, 4); l++)
			{
				int now = table[l - 1];

				z += m_mappedZ[now];
			}

			m_cgZ[mappedIndex] = z; // 最終結果
		}
}



