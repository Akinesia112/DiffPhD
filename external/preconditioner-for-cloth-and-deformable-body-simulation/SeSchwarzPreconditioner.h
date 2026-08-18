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

#pragma once

#include "SeCsr.h"
#include "SeMorton.h"
#include "SeAabbSimd.h"
#include "SeCollisionElements.h"


SE_NAMESPACE_BEGIN

// 實現 GPU-based Multilevel Additive Schwarz (MAS) 預處理器的主要類別。
class SeSchwarzPreconditioner
{

public:
	
	//==== input data

	const SeVec3fSimd* m_positions = nullptr;		// the mesh nodal positions

	//==== input mesh topology data for collision computation

	const Int4* m_edges = nullptr;				// indices of the two adjacent vertices and the two opposite vertices of edges
	const Int4* m_faces = nullptr;				// indices of the three adjacent vertices of faces, the fourth is useless
	
	const SeCsr<int>* m_neighbours = nullptr;	// the adjacent information of vertices stored in a csr format

public:

	//==== call before time integration once a frame
	// [對應論文 Section 5: Multilevel Domain Construction]
	// 初始化預處理器，包含記憶體配置與初步的空間排序。
	void AllocatePrecoditioner(int numVerts, int numEdges, int numFaces);

	//==== call before PCG iteration loop
	// [對應論文 Section 6: Matrix Precomputation]
	// [對應論文 Algorithm 1: Matrix Precomputation]
	// 在每次 PCG 迭代前呼叫，負責組裝系統矩陣並計算子區域矩陣的逆 (Inversion)。
	// 包含 PrepareCollisionHessian, PrepareHessian, LDLtInverse512 等步驟。
	void PreparePreconditioner(const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges,
		const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts);

	//==== call during PCG iterations
	// [對應論文 Section 7: Runtime Preconditioning]
	// [對應論文 Equation 5]
	// 執行預處理運算： z = M_{MAS}^{-1} * r
	// M_{MAS}^{-1} = M_{(0)}^{-1} + Sum( C_{(l)}^T * M_{(l)}^{-1} * C_{(l)} )
	void Preconditioning(SeVec3fSimd* z, const SeVec3fSimd* residual, int dim);

	//==== GPU data accessors (for uploading preconditioner data to GPU)
	const float* GetInvSymR() const { return m_invSymR.data(); }
	int GetInvSymRSize() const { return static_cast<int>(m_invSymR.size()); }
	const int* GetGoingNext() const { return m_goingNext.data(); }
	int GetGoingNextSize() const { return static_cast<int>(m_goingNext.size()); }
	const int* GetMapperSortedGetOriginal() const { return m_MapperSortedGetOriginal.data(); }
	int GetNumVerts() const { return m_numVerts; }
	int GetTotalSz() const { return m_totalSz; }
	int GetNumLevel() const { return m_numLevel; }
	int GetTotalNumberClusters() const { return m_totalNumberClusters; }

	// Flatten Int4 coarseTables to int[numVerts * 4] for GPU upload.
	std::vector<int> GetCoarseTablesFlat() const {
		std::vector<int> flat(m_numVerts * 4, 0);
		for (int i = 0; i < static_cast<int>(m_coarseTables.size()) && i < m_numVerts; i++) {
			flat[i * 4 + 0] = m_coarseTables[i][0];
			flat[i * 4 + 1] = m_coarseTables[i][1];
			flat[i * 4 + 2] = m_coarseTables[i][2];
			flat[i * 4 + 3] = m_coarseTables[i][3];
		}
		return flat;
	}

	// Extract .x (size) and .y (prefix begin) from m_levelSize for GPU upload.
	std::vector<int> GetLevelSizeX() const {
		std::vector<int> v(m_levelSize.size());
		for (size_t i = 0; i < m_levelSize.size(); i++) v[i] = m_levelSize[i][0];
		return v;
	}
	std::vector<int> GetLevelSizeY() const {
		std::vector<int> v(m_levelSize.size());
		for (size_t i = 0; i < m_levelSize.size(); i++) v[i] = m_levelSize[i][1];
		return v;
	}

private:

	int m_numVerts = 0;
	int m_numEdges = 0;
	int m_numFaces = 0;

	int m_frameIndex = 0;

	// [對應論文 Section 6.1] "In this work, we set M = 32"
	// 每個子區域 (Domain) 的大小 (bank size)。
	int m_totalSz = 0;
	// [對應論文 Section 5.2] 層級總數 (L)。
	int m_numLevel = 0;

	int m_totalNumberClusters;

	
	SeAabb<SeVec3fSimd>					m_aabb; // 用於計算 Morton Code 的邊界框


	SeArray2D<SeMatrix3f>				m_hessianMapped;
	std::vector<int>					m_mappedNeighborsNum;
	std::vector<int>					m_mappedNeighborsNumRemain;
	SeArray2D<int>						m_mappedNeighbors;
	SeArray2D<int>						m_mappedNeighborsRemain;

	// [對應論文 Section 5.2]
	// 儲存粗糙空間層級關係的表格，對應論文中的限制算子 C_(l) 或映射 map_(l)。
	SeArray2D<int>						m_CoarseSpaceTables;

	// 以下是用於層級建構 (Algorithm 2) 的輔助資料結構
	std::vector<int>                    m_prefixOrignal;
	std::vector<unsigned int>           m_fineConnectMask;  // [對應 Section 5.2] 用於判斷連接性
	std::vector<unsigned int>           m_nextConnectMsk;
	std::vector<unsigned int>           m_nextPrefix;

	std::vector<SeVec3fSimd>			m_mappedPos;


	std::vector<Int4>					m_coarseTables; // 壓縮後的層級映射表
	std::vector<int>					m_goingNext; // 指向下一層級對應節點的索引
	std::vector<int>					m_denseLevel;
	std::vector<Int2>					m_levelSize;  // .x = current level size    .y = (prefixed) current level begin index

	// 預處理過程中的暫存向量
	std::vector<SeVec3fSimd>			m_mappedR; // [對應 Equation 5] 限制後的殘差 C_(l) * r
	std::vector<SeVec3fSimd>			m_mappedZ; // [對應 Equation 5] 局部求解後的結果 y_(l)

	// [對應論文 Section 5.1: Spatial Sorting]
	// 儲存排序後的索引映射，用於確保記憶體局部性 (Locality)。
	std::vector<int>                    m_MapperSortedGetOriginal;          // sorted by morton
	std::vector<int>                    m_mapperOriginalGetSorted;

	// [對應論文 Section 5.1] "We then obtain the 60-bit Morton code..."
	std::vector<SeMorton64>             m_mortonCode;

	// [對應論文 Section 6.1: Matrix Assembly]
	// 儲存組裝後的子區域矩陣 A_{d,(l)}。
	SeArray2D<SeMatrix3f>				m_hessian32;

	// 處理碰撞產生的額外 Hessian 項
	std::vector<SeMatrix3f>             m_additionalHessian32;

	// [對應論文 Section 6.2]
	// [對應論文 Figure 11: Alternative compact format]
	// 儲存分解後或求逆後的矩陣 (L^{-T} D^{-1} L^{-1})，使用緊湊格式以節省頻寬。
	std::vector<float>                  m_invSymR;

	
	int m_stencilNum;
	int m_maxStencilNum;

	// [對應論文 Section 5.2]
	// 用於處理碰撞約束 (Collision constraints) 的模板，避免錯誤的耦合 (Artifacts)。
	std::vector<Stencil>				m_stencils;
	std::vector<Int5>					m_stencilIndexMapped;

private:

	void ComputeLevelNums(int bankSize);

	void DoAlllocation();

	// [對應論文 Section 5.1: Spatial Sorting]
	// "Similar to [Wu et al. 2015], we choose to sort the nodes by their Morton codes first."
	void ComputeTotalAABB();

	void ComputeAABB();

	void SpaceSort();

	// [對應論文 Section 5.1] 計算 Morton Codes。
	void FillSortingData();

	void DoingSort();

	void ComputeInverseMapper();

	void MapHessianTable();

	void PrepareCollisionStencils(const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts);

	void MapCollisionStencilIndices();

	// [對應論文 Section 5.2: Coarse Space Construction]
	// "construct a Nicolaides' coarse space by grouping nodes..."
	// 建構多層級結構，包含處理碰撞連接。
	void ReorderRealtime();



	void BuildConnectMaskL0();

	// [對應論文 Section 5.2] 
	// "check every domain-sized supernode... and split it... if they are not actually connected."
	// 處理碰撞導致的連接性更新。
	void BuildCollisionConnection(unsigned int* pConnect, const int* pCoarseTable);

	void PreparePrefixSumL0();

	void BuildLevel1();

	// [對應論文 Algorithm 2] 建構第 l 層的連接遮罩。
	void BuildConnectMaskLx(int level);

	void NextLevelCluster(int level);

	void PrefixSumLx(int level);

	void ComputeNextLevel(int level);

	void TotalNodes();

	// [對應論文 Section 5.2] "We collapse {map_l->l+1} to compute map_(l)..."
	// 聚合層級映射表。
	void AggregationKernel();

	void AdditionalSchwarzHessian2(SeMatrix3f hessian, std::vector<SeMatrix3f>& pAdditionalHessian, SeArray2D<SeMatrix3f>& pDenseHessian, int v1, int v2, const std::vector<int>& pGoingNext, int nLevel, int vNum);

	void PrepareCollisionHessian();

	// [對應論文 Algorithm 1: Matrix Precomputation]
	// 步驟 1-9: 組裝 Hessian 矩陣。
	void PrepareHessian(const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges);

	// [對應論文 Algorithm 1 & Section 6.2]
	// 步驟 10-12: 計算子矩陣的逆 (Fast Sub-Matrix Inversion)。
	void LDLtInverse512();

	// [對應論文 Equation 5 & Section 7]
	// 準備殘差層級 (C_(l) * r)。
	void BuildResidualHierarchy(const SeVec3fSimd* m_cgResidual);

	// [對應論文 Section 7.1: Symmetric Matrix-Vector Multiplication]
	// [對應論文 Figure 12 & 13]
	// 執行局部的矩陣向量乘法求解。
	void SchwarzLocalXSym();

	// [對應論文 Equation 5]
	// "Summing up the results at all levels into a joint output."
	void CollectFinalZ(SeVec3fSimd* m_cgZ);
};


SE_NAMESPACE_END

