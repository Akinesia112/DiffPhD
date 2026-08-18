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

#include "SeMath.h"

SE_NAMESPACE_BEGIN

/*************************************************************************
****************************    SeMorton64    ****************************
*************************************************************************/

/**
 *	@brief	64-bit Morton-Code object.
 */

// [對應論文 Section 5.1]
// 用於將 3D 座標編碼為 64-bit 整數 (Morton Code/Z-order curve)，
// 目的：將 3D 空間座標映射為 1D 整數 (Morton Code)，讓空間上相鄰的節點在記憶體中也能相鄰 (提升 Cache locality)。
// 這也是後續 "Multilevel Domain Construction" 進行分群的基礎。
class SE_ALIGN(8) SeMorton64
{

public:

	using value_type = unsigned long long;

	//!	@brief	Default constructor.
	 SeMorton64() {}

	//!	@brief	Convert to vulue_type.
	 operator value_type() const { return m_Value; }

	//!	@brief	Constructed by a given value.
	 SeMorton64(value_type _Value) : m_Value(_Value) {}

	//!	@brief	Encoded by the given 3d point located within the unit cube (controlable precision).
	template<unsigned int precision>  void Encode(float x, float y, float z, value_type lastBits)
	{
		static_assert(precision <= 21, "The highest precision for 64-bit Morton code is 21.");

		// 將浮點數座標量化為整數索引
		x = Math::Clamp(x * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);
		y = Math::Clamp(y * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);
		z = Math::Clamp(z * (1ll << precision), 0.0f, (1ll << precision) - 1.0f);

		// [對應論文 Section 5.1]
		// "interleaving the bits of its cell indices"
		// 將 x, y, z 的位元擴展並交錯。
		// 這裡實作了位元交錯邏輯：例如精確度為 21 時，每個軸佔 21 bits，共 63 bits。
		value_type xx = SeMorton64::ExpandBits(static_cast<value_type>(x)) << (66 - 3 * precision);
		value_type yy = SeMorton64::ExpandBits(static_cast<value_type>(y)) << (65 - 3 * precision);
		value_type zz = SeMorton64::ExpandBits(static_cast<value_type>(z)) << (64 - 3 * precision);

		constexpr value_type bitMask = ~value_type(0) >> (3 * precision);

		m_Value = xx + yy + zz + (lastBits & bitMask);
	}

	//!	@brief	Encoded by the given 3d point located within the unit cube (full precision).
	// Encode 函式實作了 "interleaving the bits" 的邏輯
	// [對應論文 Section 5.1]
	// "We then obtain the 60-bit Morton code of each node by simply interleaving the bits of its cell indices."
	// 論文提到 60-bit 即每軸 20 bits，此程式碼使用每軸 21 bits 達到 63-bit 更高精度，原理完全相同
	 void Encode(float x, float y, float z)
	{
		// 步驟 1: 量化 (Quantization)
		// 將 [0,1] 的浮點數座標轉換為整數網格索引 (Cell Indices)
		x = Math::Clamp(x * (1ll << 21), 0.0f, (1ll << 21) - 1.0f);
		y = Math::Clamp(y * (1ll << 21), 0.0f, (1ll << 21) - 1.0f);
		z = Math::Clamp(z * (1ll << 21), 0.0f, (1ll << 21) - 1.0f);

		// 步驟 2: 擴展位元 (Bit Expansion)
		value_type xx = SeMorton64::ExpandBits(static_cast<value_type>(x));
		value_type yy = SeMorton64::ExpandBits(static_cast<value_type>(y));
		value_type zz = SeMorton64::ExpandBits(static_cast<value_type>(z));

		// 步驟 3: 交錯位元 (Interleaving)
		// 組合結果：... Z1 Y1 X1 Z0 Y0 X0
		m_Value = (xx << 2) + (yy << 1) + zz;
	}

private:

	/**
	 *	@brief	Expand bits by inserting two zeros after each bit.
	 *	@e.g.	0000 0000 1111  ->  0010 0100 1001
	 */
	// [對應論文 Section 5.1 隱含的位元運算細節]
	// 這是實現 Morton Code 核心的 "Bit Dilation" (位元擴張) 演算法。
	// 將每個位元分開，中間插入兩個 0，以便隨後插入另外兩個軸的位元。
	// 例如：輸入二進位 11，擴展後變為 001001。
	static  value_type ExpandBits(value_type bits)
	{
		bits = (bits | (bits << 32)) & 0xFFFF00000000FFFFu;
		bits = (bits | (bits << 16)) & 0x00FF0000FF0000FFu;
		bits = (bits | (bits <<  8)) & 0xF00F00F00F00F00Fu;
		bits = (bits | (bits <<  4)) & 0x30C30C30C30C30C3u;
		return (bits | (bits <<  2)) & 0x9249249249249249u;
	}

private:

	value_type		m_Value;
};


SE_NAMESPACE_END
