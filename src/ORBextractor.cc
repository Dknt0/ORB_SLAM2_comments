/**
 * This file is part of ORB-SLAM2.
 * This file is based on the file orb.cpp from the OpenCV library (see BSD
 * license below).
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "ORBextractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace ORB_SLAM2 {

const int PATCH_SIZE = 31;       // 描述子计算范围直径
const int HALF_PATCH_SIZE = 15;  // 描述子计算范围半径
const int EDGE_THRESHOLD = 19;   // 角点提取边界   16 + 3

/// @brief 计算灰度质心角
/// @param image 图像
/// @param pt 特征点
/// @param u_max 1/4 圆弧坐标
/// @return 灰度质心角 deg
static float IC_Angle(const Mat& image, Point2f pt, const vector<int>& u_max) {
  int m_01 = 0;  // 纵向
  int m_10 = 0;  // 横向

  const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

  // Treat the center line differently, v=0
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
    m_10 += u * center[u];

  // Go line by line in the circuI853lar patch
  int step = (int)image.step1();  // 图像行宽
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
    // Proceed over the two lines
    int v_sum = 0;  // 一行的灰度和
    int d = u_max[v];
    for (int u = -d; u <= d; ++u) {
      int val_plus = center[u + v * step];   // 上
      int val_minus = center[u - v * step];  // 下
      v_sum += (val_plus - val_minus);       // 用于计算 v 矩
      m_10 += u * (val_plus + val_minus);    // u 矩
    }
    m_01 += v * v_sum;  // v 矩
  }

  return fastAtan2((float)m_01, (float)m_10);  // 返回角度
}

// 角度转弧度
const float factorPI = (float)(CV_PI / 180.f);

/// @brief 计算单个 BRIEF 描述子
/// @param kpt 特征点
/// @param img 图像
/// @param pattern 模板
/// @param desc 描述子
static void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img,
                                 const Point* pattern, uchar* desc) {
  float angle = (float)kpt.angle * factorPI;  // 主方向角度 弧度
  float a = (float)cos(angle), b = (float)sin(angle);

  const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  const int step = (int)img.step;  // 图像每一行占用的比特数

/// 第 idx 个位置模板  旋转后在图像中对应像素值
#define GET_VALUE(idx)                                             \
  center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
         cvRound(pattern[idx].x * a - pattern[idx].y * b)]

  // 8 * 32 = 256 位描述子
  // 8 * 2 = 16   一次循环使用 8 对模板点
  for (int i = 0; i < 32; ++i, pattern += 16) {
    int t0, t1, val;
    t0 = GET_VALUE(0);
    t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2);
    t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    desc[i] = (uchar)val;
  }

#undef GET_VALUE
}

/// @brief BRIEF 描述子计算模板，机器学习生成的
static int bit_pattern_31_[256 * 4] = {
    8,   -3,  9,   5 /*mean (0), correlation (0)*/,
    4,   2,   7,   -12 /*mean (1.12461e-05), correlation (0.0437584)*/,
    -11, 9,   -8,  2 /*mean (3.37382e-05), correlation (0.0617409)*/,
    7,   -12, 12,  -13 /*mean (5.62303e-05), correlation (0.0636977)*/,
    2,   -13, 2,   12 /*mean (0.000134953), correlation (0.085099)*/,
    1,   -7,  1,   6 /*mean (0.000528565), correlation (0.0857175)*/,
    -2,  -10, -2,  -4 /*mean (0.0188821), correlation (0.0985774)*/,
    -13, -13, -11, -8 /*mean (0.0363135), correlation (0.0899616)*/,
    -13, -3,  -12, -9 /*mean (0.121806), correlation (0.099849)*/,
    10,  4,   11,  9 /*mean (0.122065), correlation (0.093285)*/,
    -13, -8,  -8,  -9 /*mean (0.162787), correlation (0.0942748)*/,
    -11, 7,   -9,  12 /*mean (0.21561), correlation (0.0974438)*/,
    7,   7,   12,  6 /*mean (0.160583), correlation (0.130064)*/,
    -4,  -5,  -3,  0 /*mean (0.228171), correlation (0.132998)*/,
    -13, 2,   -12, -3 /*mean (0.00997526), correlation (0.145926)*/,
    -9,  0,   -7,  5 /*mean (0.198234), correlation (0.143636)*/,
    12,  -6,  12,  -1 /*mean (0.0676226), correlation (0.16689)*/,
    -3,  6,   -2,  12 /*mean (0.166847), correlation (0.171682)*/,
    -6,  -13, -4,  -8 /*mean (0.101215), correlation (0.179716)*/,
    11,  -13, 12,  -8 /*mean (0.200641), correlation (0.192279)*/,
    4,   7,   5,   1 /*mean (0.205106), correlation (0.186848)*/,
    5,   -3,  10,  -3 /*mean (0.234908), correlation (0.192319)*/,
    3,   -7,  6,   12 /*mean (0.0709964), correlation (0.210872)*/,
    -8,  -7,  -6,  -2 /*mean (0.0939834), correlation (0.212589)*/,
    -2,  11,  -1,  -10 /*mean (0.127778), correlation (0.20866)*/,
    -13, 12,  -8,  10 /*mean (0.14783), correlation (0.206356)*/,
    -7,  3,   -5,  -3 /*mean (0.182141), correlation (0.198942)*/,
    -4,  2,   -3,  7 /*mean (0.188237), correlation (0.21384)*/,
    -10, -12, -6,  11 /*mean (0.14865), correlation (0.23571)*/,
    5,   -12, 6,   -7 /*mean (0.222312), correlation (0.23324)*/,
    5,   -6,  7,   -1 /*mean (0.229082), correlation (0.23389)*/,
    1,   0,   4,   -5 /*mean (0.241577), correlation (0.215286)*/,
    9,   11,  11,  -13 /*mean (0.00338507), correlation (0.251373)*/,
    4,   7,   4,   12 /*mean (0.131005), correlation (0.257622)*/,
    2,   -1,  4,   4 /*mean (0.152755), correlation (0.255205)*/,
    -4,  -12, -2,  7 /*mean (0.182771), correlation (0.244867)*/,
    -8,  -5,  -7,  -10 /*mean (0.186898), correlation (0.23901)*/,
    4,   11,  9,   12 /*mean (0.226226), correlation (0.258255)*/,
    0,   -8,  1,   -13 /*mean (0.0897886), correlation (0.274827)*/,
    -13, -2,  -8,  2 /*mean (0.148774), correlation (0.28065)*/,
    -3,  -2,  -2,  3 /*mean (0.153048), correlation (0.283063)*/,
    -6,  9,   -4,  -9 /*mean (0.169523), correlation (0.278248)*/,
    8,   12,  10,  7 /*mean (0.225337), correlation (0.282851)*/,
    0,   9,   1,   3 /*mean (0.226687), correlation (0.278734)*/,
    7,   -5,  11,  -10 /*mean (0.00693882), correlation (0.305161)*/,
    -13, -6,  -11, 0 /*mean (0.0227283), correlation (0.300181)*/,
    10,  7,   12,  1 /*mean (0.125517), correlation (0.31089)*/,
    -6,  -3,  -6,  12 /*mean (0.131748), correlation (0.312779)*/,
    10,  -9,  12,  -4 /*mean (0.144827), correlation (0.292797)*/,
    -13, 8,   -8,  -12 /*mean (0.149202), correlation (0.308918)*/,
    -13, 0,   -8,  -4 /*mean (0.160909), correlation (0.310013)*/,
    3,   3,   7,   8 /*mean (0.177755), correlation (0.309394)*/,
    5,   7,   10,  -7 /*mean (0.212337), correlation (0.310315)*/,
    -1,  7,   1,   -12 /*mean (0.214429), correlation (0.311933)*/,
    3,   -10, 5,   6 /*mean (0.235807), correlation (0.313104)*/,
    2,   -4,  3,   -10 /*mean (0.00494827), correlation (0.344948)*/,
    -13, 0,   -13, 5 /*mean (0.0549145), correlation (0.344675)*/,
    -13, -7,  -12, 12 /*mean (0.103385), correlation (0.342715)*/,
    -13, 3,   -11, 8 /*mean (0.134222), correlation (0.322922)*/,
    -7,  12,  -4,  7 /*mean (0.153284), correlation (0.337061)*/,
    6,   -10, 12,  8 /*mean (0.154881), correlation (0.329257)*/,
    -9,  -1,  -7,  -6 /*mean (0.200967), correlation (0.33312)*/,
    -2,  -5,  0,   12 /*mean (0.201518), correlation (0.340635)*/,
    -12, 5,   -7,  5 /*mean (0.207805), correlation (0.335631)*/,
    3,   -10, 8,   -13 /*mean (0.224438), correlation (0.34504)*/,
    -7,  -7,  -4,  5 /*mean (0.239361), correlation (0.338053)*/,
    -3,  -2,  -1,  -7 /*mean (0.240744), correlation (0.344322)*/,
    2,   9,   5,   -11 /*mean (0.242949), correlation (0.34145)*/,
    -11, -13, -5,  -13 /*mean (0.244028), correlation (0.336861)*/,
    -1,  6,   0,   -1 /*mean (0.247571), correlation (0.343684)*/,
    5,   -3,  5,   2 /*mean (0.000697256), correlation (0.357265)*/,
    -4,  -13, -4,  12 /*mean (0.00213675), correlation (0.373827)*/,
    -9,  -6,  -9,  6 /*mean (0.0126856), correlation (0.373938)*/,
    -12, -10, -8,  -4 /*mean (0.0152497), correlation (0.364237)*/,
    10,  2,   12,  -3 /*mean (0.0299933), correlation (0.345292)*/,
    7,   12,  12,  12 /*mean (0.0307242), correlation (0.366299)*/,
    -7,  -13, -6,  5 /*mean (0.0534975), correlation (0.368357)*/,
    -4,  9,   -3,  4 /*mean (0.099865), correlation (0.372276)*/,
    7,   -1,  12,  2 /*mean (0.117083), correlation (0.364529)*/,
    -7,  6,   -5,  1 /*mean (0.126125), correlation (0.369606)*/,
    -13, 11,  -12, 5 /*mean (0.130364), correlation (0.358502)*/,
    -3,  7,   -2,  -6 /*mean (0.131691), correlation (0.375531)*/,
    7,   -8,  12,  -7 /*mean (0.160166), correlation (0.379508)*/,
    -13, -7,  -11, -12 /*mean (0.167848), correlation (0.353343)*/,
    1,   -3,  12,  12 /*mean (0.183378), correlation (0.371916)*/,
    2,   -6,  3,   0 /*mean (0.228711), correlation (0.371761)*/,
    -4,  3,   -2,  -13 /*mean (0.247211), correlation (0.364063)*/,
    -1,  -13, 1,   9 /*mean (0.249325), correlation (0.378139)*/,
    7,   1,   8,   -6 /*mean (0.000652272), correlation (0.411682)*/,
    1,   -1,  3,   12 /*mean (0.00248538), correlation (0.392988)*/,
    9,   1,   12,  6 /*mean (0.0206815), correlation (0.386106)*/,
    -1,  -9,  -1,  3 /*mean (0.0364485), correlation (0.410752)*/,
    -13, -13, -10, 5 /*mean (0.0376068), correlation (0.398374)*/,
    7,   7,   10,  12 /*mean (0.0424202), correlation (0.405663)*/,
    12,  -5,  12,  9 /*mean (0.0942645), correlation (0.410422)*/,
    6,   3,   7,   11 /*mean (0.1074), correlation (0.413224)*/,
    5,   -13, 6,   10 /*mean (0.109256), correlation (0.408646)*/,
    2,   -12, 2,   3 /*mean (0.131691), correlation (0.416076)*/,
    3,   8,   4,   -6 /*mean (0.165081), correlation (0.417569)*/,
    2,   6,   12,  -13 /*mean (0.171874), correlation (0.408471)*/,
    9,   -12, 10,  3 /*mean (0.175146), correlation (0.41296)*/,
    -8,  4,   -7,  9 /*mean (0.183682), correlation (0.402956)*/,
    -11, 12,  -4,  -6 /*mean (0.184672), correlation (0.416125)*/,
    1,   12,  2,   -8 /*mean (0.191487), correlation (0.386696)*/,
    6,   -9,  7,   -4 /*mean (0.192668), correlation (0.394771)*/,
    2,   3,   3,   -2 /*mean (0.200157), correlation (0.408303)*/,
    6,   3,   11,  0 /*mean (0.204588), correlation (0.411762)*/,
    3,   -3,  8,   -8 /*mean (0.205904), correlation (0.416294)*/,
    7,   8,   9,   3 /*mean (0.213237), correlation (0.409306)*/,
    -11, -5,  -6,  -4 /*mean (0.243444), correlation (0.395069)*/,
    -10, 11,  -5,  10 /*mean (0.247672), correlation (0.413392)*/,
    -5,  -8,  -3,  12 /*mean (0.24774), correlation (0.411416)*/,
    -10, 5,   -9,  0 /*mean (0.00213675), correlation (0.454003)*/,
    8,   -1,  12,  -6 /*mean (0.0293635), correlation (0.455368)*/,
    4,   -6,  6,   -11 /*mean (0.0404971), correlation (0.457393)*/,
    -10, 12,  -8,  7 /*mean (0.0481107), correlation (0.448364)*/,
    4,   -2,  6,   7 /*mean (0.050641), correlation (0.455019)*/,
    -2,  0,   -2,  12 /*mean (0.0525978), correlation (0.44338)*/,
    -5,  -8,  -5,  2 /*mean (0.0629667), correlation (0.457096)*/,
    7,   -6,  10,  12 /*mean (0.0653846), correlation (0.445623)*/,
    -9,  -13, -8,  -8 /*mean (0.0858749), correlation (0.449789)*/,
    -5,  -13, -5,  -2 /*mean (0.122402), correlation (0.450201)*/,
    8,   -8,  9,   -13 /*mean (0.125416), correlation (0.453224)*/,
    -9,  -11, -9,  0 /*mean (0.130128), correlation (0.458724)*/,
    1,   -8,  1,   -2 /*mean (0.132467), correlation (0.440133)*/,
    7,   -4,  9,   1 /*mean (0.132692), correlation (0.454)*/,
    -2,  1,   -1,  -4 /*mean (0.135695), correlation (0.455739)*/,
    11,  -6,  12,  -11 /*mean (0.142904), correlation (0.446114)*/,
    -12, -9,  -6,  4 /*mean (0.146165), correlation (0.451473)*/,
    3,   7,   7,   12 /*mean (0.147627), correlation (0.456643)*/,
    5,   5,   10,  8 /*mean (0.152901), correlation (0.455036)*/,
    0,   -4,  2,   8 /*mean (0.167083), correlation (0.459315)*/,
    -9,  12,  -5,  -13 /*mean (0.173234), correlation (0.454706)*/,
    0,   7,   2,   12 /*mean (0.18312), correlation (0.433855)*/,
    -1,  2,   1,   7 /*mean (0.185504), correlation (0.443838)*/,
    5,   11,  7,   -9 /*mean (0.185706), correlation (0.451123)*/,
    3,   5,   6,   -8 /*mean (0.188968), correlation (0.455808)*/,
    -13, -4,  -8,  9 /*mean (0.191667), correlation (0.459128)*/,
    -5,  9,   -3,  -3 /*mean (0.193196), correlation (0.458364)*/,
    -4,  -7,  -3,  -12 /*mean (0.196536), correlation (0.455782)*/,
    6,   5,   8,   0 /*mean (0.1972), correlation (0.450481)*/,
    -7,  6,   -6,  12 /*mean (0.199438), correlation (0.458156)*/,
    -13, 6,   -5,  -2 /*mean (0.211224), correlation (0.449548)*/,
    1,   -10, 3,   10 /*mean (0.211718), correlation (0.440606)*/,
    4,   1,   8,   -4 /*mean (0.213034), correlation (0.443177)*/,
    -2,  -2,  2,   -13 /*mean (0.234334), correlation (0.455304)*/,
    2,   -12, 12,  12 /*mean (0.235684), correlation (0.443436)*/,
    -2,  -13, 0,   -6 /*mean (0.237674), correlation (0.452525)*/,
    4,   1,   9,   3 /*mean (0.23962), correlation (0.444824)*/,
    -6,  -10, -3,  -5 /*mean (0.248459), correlation (0.439621)*/,
    -3,  -13, -1,  1 /*mean (0.249505), correlation (0.456666)*/,
    7,   5,   12,  -11 /*mean (0.00119208), correlation (0.495466)*/,
    4,   -2,  5,   -7 /*mean (0.00372245), correlation (0.484214)*/,
    -13, 9,   -9,  -5 /*mean (0.00741116), correlation (0.499854)*/,
    7,   1,   8,   6 /*mean (0.0208952), correlation (0.499773)*/,
    7,   -8,  7,   6 /*mean (0.0220085), correlation (0.501609)*/,
    -7,  -4,  -7,  1 /*mean (0.0233806), correlation (0.496568)*/,
    -8,  11,  -7,  -8 /*mean (0.0236505), correlation (0.489719)*/,
    -13, 6,   -12, -8 /*mean (0.0268781), correlation (0.503487)*/,
    2,   4,   3,   9 /*mean (0.0323324), correlation (0.501938)*/,
    10,  -5,  12,  3 /*mean (0.0399235), correlation (0.494029)*/,
    -6,  -5,  -6,  7 /*mean (0.0420153), correlation (0.486579)*/,
    8,   -3,  9,   -8 /*mean (0.0548021), correlation (0.484237)*/,
    2,   -12, 2,   8 /*mean (0.0616622), correlation (0.496642)*/,
    -11, -2,  -10, 3 /*mean (0.0627755), correlation (0.498563)*/,
    -12, -13, -7,  -9 /*mean (0.0829622), correlation (0.495491)*/,
    -11, 0,   -10, -5 /*mean (0.0843342), correlation (0.487146)*/,
    5,   -3,  11,  8 /*mean (0.0929937), correlation (0.502315)*/,
    -2,  -13, -1,  12 /*mean (0.113327), correlation (0.48941)*/,
    -1,  -8,  0,   9 /*mean (0.132119), correlation (0.467268)*/,
    -13, -11, -12, -5 /*mean (0.136269), correlation (0.498771)*/,
    -10, -2,  -10, 11 /*mean (0.142173), correlation (0.498714)*/,
    -3,  9,   -2,  -13 /*mean (0.144141), correlation (0.491973)*/,
    2,   -3,  3,   2 /*mean (0.14892), correlation (0.500782)*/,
    -9,  -13, -4,  0 /*mean (0.150371), correlation (0.498211)*/,
    -4,  6,   -3,  -10 /*mean (0.152159), correlation (0.495547)*/,
    -4,  12,  -2,  -7 /*mean (0.156152), correlation (0.496925)*/,
    -6,  -11, -4,  9 /*mean (0.15749), correlation (0.499222)*/,
    6,   -3,  6,   11 /*mean (0.159211), correlation (0.503821)*/,
    -13, 11,  -5,  5 /*mean (0.162427), correlation (0.501907)*/,
    11,  11,  12,  6 /*mean (0.16652), correlation (0.497632)*/,
    7,   -5,  12,  -2 /*mean (0.169141), correlation (0.484474)*/,
    -1,  12,  0,   7 /*mean (0.169456), correlation (0.495339)*/,
    -4,  -8,  -3,  -2 /*mean (0.171457), correlation (0.487251)*/,
    -7,  1,   -6,  7 /*mean (0.175), correlation (0.500024)*/,
    -13, -12, -8,  -13 /*mean (0.175866), correlation (0.497523)*/,
    -7,  -2,  -6,  -8 /*mean (0.178273), correlation (0.501854)*/,
    -8,  5,   -6,  -9 /*mean (0.181107), correlation (0.494888)*/,
    -5,  -1,  -4,  5 /*mean (0.190227), correlation (0.482557)*/,
    -13, 7,   -8,  10 /*mean (0.196739), correlation (0.496503)*/,
    1,   5,   5,   -13 /*mean (0.19973), correlation (0.499759)*/,
    1,   0,   10,  -13 /*mean (0.204465), correlation (0.49873)*/,
    9,   12,  10,  -1 /*mean (0.209334), correlation (0.49063)*/,
    5,   -8,  10,  -9 /*mean (0.211134), correlation (0.503011)*/,
    -1,  11,  1,   -13 /*mean (0.212), correlation (0.499414)*/,
    -9,  -3,  -6,  2 /*mean (0.212168), correlation (0.480739)*/,
    -1,  -10, 1,   12 /*mean (0.212731), correlation (0.502523)*/,
    -13, 1,   -8,  -10 /*mean (0.21327), correlation (0.489786)*/,
    8,   -11, 10,  -6 /*mean (0.214159), correlation (0.488246)*/,
    2,   -13, 3,   -6 /*mean (0.216993), correlation (0.50287)*/,
    7,   -13, 12,  -9 /*mean (0.223639), correlation (0.470502)*/,
    -10, -10, -5,  -7 /*mean (0.224089), correlation (0.500852)*/,
    -10, -8,  -8,  -13 /*mean (0.228666), correlation (0.502629)*/,
    4,   -6,  8,   5 /*mean (0.22906), correlation (0.498305)*/,
    3,   12,  8,   -13 /*mean (0.233378), correlation (0.503825)*/,
    -4,  2,   -3,  -3 /*mean (0.234323), correlation (0.476692)*/,
    5,   -13, 10,  -12 /*mean (0.236392), correlation (0.475462)*/,
    4,   -13, 5,   -1 /*mean (0.236842), correlation (0.504132)*/,
    -9,  9,   -4,  3 /*mean (0.236977), correlation (0.497739)*/,
    0,   3,   3,   -9 /*mean (0.24314), correlation (0.499398)*/,
    -12, 1,   -6,  1 /*mean (0.243297), correlation (0.489447)*/,
    3,   2,   4,   -8 /*mean (0.00155196), correlation (0.553496)*/,
    -10, -10, -10, 9 /*mean (0.00239541), correlation (0.54297)*/,
    8,   -13, 12,  12 /*mean (0.0034413), correlation (0.544361)*/,
    -8,  -12, -6,  -5 /*mean (0.003565), correlation (0.551225)*/,
    2,   2,   3,   7 /*mean (0.00835583), correlation (0.55285)*/,
    10,  6,   11,  -8 /*mean (0.00885065), correlation (0.540913)*/,
    6,   8,   8,   -12 /*mean (0.0101552), correlation (0.551085)*/,
    -7,  10,  -6,  5 /*mean (0.0102227), correlation (0.533635)*/,
    -3,  -9,  -3,  9 /*mean (0.0110211), correlation (0.543121)*/,
    -1,  -13, -1,  5 /*mean (0.0113473), correlation (0.550173)*/,
    -3,  -7,  -3,  4 /*mean (0.0140913), correlation (0.554774)*/,
    -8,  -2,  -8,  3 /*mean (0.017049), correlation (0.55461)*/,
    4,   2,   12,  12 /*mean (0.01778), correlation (0.546921)*/,
    2,   -5,  3,   11 /*mean (0.0224022), correlation (0.549667)*/,
    6,   -9,  11,  -13 /*mean (0.029161), correlation (0.546295)*/,
    3,   -1,  7,   12 /*mean (0.0303081), correlation (0.548599)*/,
    11,  -1,  12,  4 /*mean (0.0355151), correlation (0.523943)*/,
    -3,  0,   -3,  6 /*mean (0.0417904), correlation (0.543395)*/,
    4,   -11, 4,   12 /*mean (0.0487292), correlation (0.542818)*/,
    2,   -4,  2,   1 /*mean (0.0575124), correlation (0.554888)*/,
    -10, -6,  -8,  1 /*mean (0.0594242), correlation (0.544026)*/,
    -13, 7,   -11, 1 /*mean (0.0597391), correlation (0.550524)*/,
    -13, 12,  -11, -13 /*mean (0.0608974), correlation (0.55383)*/,
    6,   0,   11,  -13 /*mean (0.065126), correlation (0.552006)*/,
    0,   -1,  1,   4 /*mean (0.074224), correlation (0.546372)*/,
    -13, 3,   -9,  -2 /*mean (0.0808592), correlation (0.554875)*/,
    -9,  8,   -6,  -3 /*mean (0.0883378), correlation (0.551178)*/,
    -13, -6,  -8,  -2 /*mean (0.0901035), correlation (0.548446)*/,
    5,   -9,  8,   10 /*mean (0.0949843), correlation (0.554694)*/,
    2,   7,   3,   -9 /*mean (0.0994152), correlation (0.550979)*/,
    -1,  -6,  -1,  -1 /*mean (0.10045), correlation (0.552714)*/,
    9,   5,   11,  -2 /*mean (0.100686), correlation (0.552594)*/,
    11,  -3,  12,  -8 /*mean (0.101091), correlation (0.532394)*/,
    3,   0,   3,   5 /*mean (0.101147), correlation (0.525576)*/,
    -1,  4,   0,   10 /*mean (0.105263), correlation (0.531498)*/,
    3,   -6,  4,   5 /*mean (0.110785), correlation (0.540491)*/,
    -13, 0,   -10, 5 /*mean (0.112798), correlation (0.536582)*/,
    5,   8,   12,  11 /*mean (0.114181), correlation (0.555793)*/,
    8,   9,   9,   -6 /*mean (0.117431), correlation (0.553763)*/,
    7,   -4,  8,   -12 /*mean (0.118522), correlation (0.553452)*/,
    -10, 4,   -10, 9 /*mean (0.12094), correlation (0.554785)*/,
    7,   3,   12,  4 /*mean (0.122582), correlation (0.555825)*/,
    9,   -7,  10,  -2 /*mean (0.124978), correlation (0.549846)*/,
    7,   0,   12,  -2 /*mean (0.127002), correlation (0.537452)*/,
    -1,  -6,  0,   -11 /*mean (0.127148), correlation (0.547401)*/
};

/// @brief ORB 提取器构造函数
/// @param _nfeatures 特征点数量
/// @param _scaleFactor 金字塔层缩放比例  大于 1
/// @param _nlevels 金字塔层数
/// @param _iniThFAST 初始阈值
/// @param _minThFAST 最小阈值
ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST)
    : nfeatures(_nfeatures),
      scaleFactor(_scaleFactor),
      nlevels(_nlevels),
      iniThFAST(_iniThFAST),
      minThFAST(_minThFAST) {
  // 计算尺寸、面积比例
  mvScaleFactor.resize(nlevels);
  mvLevelSigma2.resize(nlevels);
  mvScaleFactor[0] = 1.0f;
  mvLevelSigma2[0] = 1.0f;
  for (int i = 1; i < nlevels; i++) {
    mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
    mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
  }

  // 计算尺寸、面积逆比例
  mvInvScaleFactor.resize(nlevels);
  mvInvLevelSigma2.resize(nlevels);
  for (int i = 0; i < nlevels; i++) {
    mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
    mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
  }

  mvImagePyramid.resize(nlevels);  // 金字塔

  /// 计算每层的特征点数
  mnFeaturesPerLevel.resize(nlevels);  // 每层特征点数
  float factor = 1.0f / scaleFactor;   // 上层相对下层的尺寸，小于 1
  // 每层中提取的特征点数与尺寸成正比，注意是尺寸，不是面积。
  // 下面这个表达式利用等比数列求和公式求首项，即最底层图片中特征点的数目
  float nDesiredFeaturesPerScale =
      nfeatures * (1 - factor) /
      (1 - (float)pow((double)factor, (double)nlevels));

  int sumFeatures = 0;
  for (int level = 0; level < nlevels - 1; level++) {
    mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
    sumFeatures += mnFeaturesPerLevel[level];
    nDesiredFeaturesPerScale *= factor;
  }
  mnFeaturesPerLevel[nlevels - 1] =
      std::max(nfeatures - sumFeatures, 0);  // 保证不超过总数

  const int npoints = 512;  // 模板中的点数
  const Point* pattern0 = (const Point*)bit_pattern_31_;
  // std::back_inserter 是用于在容器末尾添加元素的迭代器
  // 下面的表达式从 bit_pattern_31_ 拷贝创建了 pattern 匹配模板
  std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

  // This is for orientation
  // pre-compute the end of a row in a circular patch
  // 提前计算描述子计算范围边界序号，方便计算 BRIEF 描述子
  // 这是个 1/4 圆弧
  umax.resize(HALF_PATCH_SIZE + 1);

  int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);  // 大
  int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);              // 小
  const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;  // 搜索半径平方
  // 这里计算了 1/8 个圆弧
  for (v = 0; v <= vmax; ++v)
    umax[v] = cvRound(sqrt(hp2 - v * v));  // 勾股定理，计算纵坐标

  // Make sure we are symmetric
  // 保证对称性，为啥能呢，这里需要做个实验测试下
  for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
    while (umax[v0] == umax[v0 + 1]) ++v0;
    umax[v] = v0;
    ++v0;
  }
}

/// @brief 计算每个特征点的主方向
/// @param image 图像
/// @param keypoints 关键点集合
/// @param umax 1/4 半圆坐标
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints,
                               const vector<int>& umax) {
  for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                  keypointEnd = keypoints.end();
       keypoint != keypointEnd; ++keypoint) {
    keypoint->angle = IC_Angle(image, keypoint->pt, umax);  // 计算特征主方向
  }
}

/// @brief 划分节点区域为四个子区域
/// @param n1 左上子块
/// @param n2 右上子块
/// @param n3 左下子块
/// @param n4 右下子块
void ExtractorNode::DivideNode(ExtractorNode& n1, ExtractorNode& n2,
                               ExtractorNode& n3, ExtractorNode& n4) {
  /// 向上取整，获得中点坐标
  const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
  const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

  // Define boundaries of childs
  /// 左上子块
  n1.UL = UL;
  n1.UR = cv::Point2i(UL.x + halfX, UL.y);
  n1.BL = cv::Point2i(UL.x, UL.y + halfY);
  n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
  n1.vKeys.reserve(vKeys.size());  // 预分配和父节点相同的空间

  /// 右上子块
  n2.UL = n1.UR;
  n2.UR = UR;
  n2.BL = n1.BR;
  n2.BR = cv::Point2i(UR.x, UL.y + halfY);
  n2.vKeys.reserve(vKeys.size());

  /// 左下子块
  n3.UL = n1.BL;
  n3.UR = n1.BR;
  n3.BL = BL;
  n3.BR = cv::Point2i(n1.BR.x, BL.y);
  n3.vKeys.reserve(vKeys.size());

  /// 右下子块
  n4.UL = n3.UR;
  n4.UR = n2.BR;
  n4.BL = n3.BR;
  n4.BR = BR;
  n4.vKeys.reserve(vKeys.size());

  // Associate points to childs
  /// 遍历，将特征点分配到对应子区域中
  for (size_t i = 0; i < vKeys.size(); i++) {
    const cv::KeyPoint& kp = vKeys[i];
    if (kp.pt.x < n1.UR.x) {
      if (kp.pt.y < n1.BR.y)
        n1.vKeys.push_back(kp);
      else
        n3.vKeys.push_back(kp);
    } else if (kp.pt.y < n1.BR.y)
      n2.vKeys.push_back(kp);
    else
      n4.vKeys.push_back(kp);
  }

  // 如果子区域只有一个特征点，则不再分割
  if (n1.vKeys.size() == 1) n1.bNoMore = true;
  if (n2.vKeys.size() == 1) n2.bNoMore = true;
  if (n3.vKeys.size() == 1) n3.bNoMore = true;
  if (n4.vKeys.size() == 1) n4.bNoMore = true;
}

/// @brief 四叉树分割
/// @param vToDistributeKeys 待分配特征点集合
/// @param minX 左边界
/// @param maxX 右边界
/// @param minY 上边界
/// @param maxY 下边界
/// @param N 目标特征点总数
/// @param level 金字塔层数
/// @return
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(
    const vector<cv::KeyPoint>& vToDistributeKeys, const int& minX,
    const int& maxX, const int& minY, const int& maxY, const int& N,
    const int& level) {
  // Compute how many initial nodes
  // 初始节点数为宽高比，使节点接近于正方形
  // 由此，图像宽必须大于高
  const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

  // 初始节点节点宽度
  const float hX = static_cast<float>(maxX - minX) / nIni;

  // 节点列表   用链表，不用向量，避免动态分配空间
  list<ExtractorNode> lNodes;

  // 初始节点指针向量
  vector<ExtractorNode*> vpIniNodes;
  vpIniNodes.resize(nIni);

  // 划分初始节点，竖着分
  for (int i = 0; i < nIni; i++) {
    ExtractorNode ni;
    // 这些坐标是在相对于提取边界左上角的坐标
    ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);      // 左上
    ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);  // 右上
    ni.BL = cv::Point2i(ni.UL.x, maxY - minY);               // 左下
    ni.BR = cv::Point2i(ni.UR.x, maxY - minY);               // 右下
    ni.vKeys.reserve(vToDistributeKeys.size());

    lNodes.push_back(ni);
    vpIniNodes[i] = &lNodes.back();
  }

  // Associate points to childs
  //  将特征点分配给子节点
  for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
    const cv::KeyPoint& kp = vToDistributeKeys[i];
    // 按照 x 坐标分配
    vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
  }

  list<ExtractorNode>::iterator lit = lNodes.begin();

  // 计算初始节点标志位
  while (lit != lNodes.end()) {
    if (lit->vKeys.size() == 1) {
      lit->bNoMore = true;
      lit++;
    } else if (lit->vKeys.empty())
      lit = lNodes.erase(lit);
    else
      lit++;
  }

  bool bFinish = false;  // 停止分配标志位

  int iteration = 0;  // 迭代次数

  vector<pair<int, ExtractorNode*> >
      vSizeAndPointerToNode;  // 节点包含特征点数量，指向节点的指针
  vSizeAndPointerToNode.reserve(lNodes.size() * 4);

  while (!bFinish) {
    iteration++;

    int prevSize = lNodes.size();  // 上一次节点列表大小

    lit = lNodes.begin();  // 列表迭代器

    int nToExpand = 0;  // 需要分割的节点数

    vSizeAndPointerToNode.clear();

    while (lit != lNodes.end()) {
      if (lit->bNoMore) {
        // If node only contains one point do not subdivide and continue
        // 仅包含一个特征点，跳过
        lit++;
        continue;
      } else {
        // If more than one point, subdivide
        // 多余一个特征点，分割
        ExtractorNode n1, n2, n3, n4;
        lit->DivideNode(n1, n2, n3, n4);

        // Add childs if they contain points
        // 如果子节点包含特征点，则添加，否则弃用
        if (n1.vKeys.size() > 0) {
          lNodes.push_front(n1);  // 添加到列表头
          if (n1.vKeys.size() > 1) {
            // 包含特征点大于一，需要进一步分割
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                make_pair(n1.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();  // 指向自己？
          }
        }
        if (n2.vKeys.size() > 0) {
          lNodes.push_front(n2);
          if (n2.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                make_pair(n2.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n3.vKeys.size() > 0) {
          lNodes.push_front(n3);
          if (n3.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                make_pair(n3.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }
        if (n4.vKeys.size() > 0) {
          lNodes.push_front(n4);
          if (n4.vKeys.size() > 1) {
            nToExpand++;
            vSizeAndPointerToNode.push_back(
                make_pair(n4.vKeys.size(), &lNodes.front()));
            lNodes.front().lit = lNodes.begin();
          }
        }

        lit = lNodes.erase(lit);  // 分割后删除父节点
        continue;
      }
    }

    // Finish if there are more nodes than required features
    // or all nodes contain just one point
    // 如果节点多于需要特征点数，或所有节点都只包含一个特征点，停止
    if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize) {
      bFinish = true;
    } else if (((int)lNodes.size() + nToExpand * 3) > N) {
      // 如果进一步分割后节点数可能多于需需要特征点数
      // 用另一种方法分割
      while (!bFinish) {
        prevSize = lNodes.size();  // 上一次节点列表大小

        vector<pair<int, ExtractorNode*> > vPrevSizeAndPointerToNode =
            vSizeAndPointerToNode;  // 上一次节点包含特征点数量，指向节点的指针
        vSizeAndPointerToNode.clear();

        // 按照节点包含特征点数量排序
        sort(vPrevSizeAndPointerToNode.begin(),
             vPrevSizeAndPointerToNode.end());
        for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
          ExtractorNode n1, n2, n3, n4;
          vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

          // Add childs if they contain points
          if (n1.vKeys.size() > 0) {
            lNodes.push_front(n1);
            if (n1.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  make_pair(n1.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n2.vKeys.size() > 0) {
            lNodes.push_front(n2);
            if (n2.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  make_pair(n2.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n3.vKeys.size() > 0) {
            lNodes.push_front(n3);
            if (n3.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  make_pair(n3.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }
          if (n4.vKeys.size() > 0) {
            lNodes.push_front(n4);
            if (n4.vKeys.size() > 1) {
              vSizeAndPointerToNode.push_back(
                  make_pair(n4.vKeys.size(), &lNodes.front()));
              lNodes.front().lit = lNodes.begin();
            }
          }

          lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

          // 如果节点数量多于目标特征点数量，结束
          if ((int)lNodes.size() >= N) break;
        }

        // 如果节点多于需要特征点数，或所有节点都只包含一个特征点，停止
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
          bFinish = true;
      }
    }
  }

  // Retain the best point in each node
  // 保留每个节点中响应最强的特征点
  vector<cv::KeyPoint> vResultKeys;
  vResultKeys.reserve(nfeatures);
  for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end();
       lit++) {
    vector<cv::KeyPoint>& vNodeKeys = lit->vKeys;
    cv::KeyPoint* pKP = &vNodeKeys[0];
    float maxResponse = pKP->response;

    // 筛选最好特征点
    for (size_t k = 1; k < vNodeKeys.size(); k++) {
      if (vNodeKeys[k].response > maxResponse) {
        pKP = &vNodeKeys[k];
        maxResponse = vNodeKeys[k].response;
      }
    }

    vResultKeys.push_back(*pKP);
  }

  return vResultKeys;
}

/// @brief 提取特征点  四叉树法
/// @param allKeypoints 每一层的特征点
void ORBextractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint> >& allKeypoints) {
  allKeypoints.resize(nlevels);  // 按层数分配空间

  const float W = 30;  // 将图像分为 30*30 大小的网格

  // 遍历金字塔所有层
  for (int level = 0; level < nlevels; ++level) {
    // 角点提取边界
    const int minBorderX = EDGE_THRESHOLD - 3;  // 左
    const int minBorderY = minBorderX;          // 上
    const int maxBorderX =
        mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;  // 右
    const int maxBorderY =
        mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;  // 下

    // 待分配特征点集合
    vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(nfeatures * 10);

    // 提取区域宽、高
    const float width = (maxBorderX - minBorderX);
    const float height = (maxBorderY - minBorderY);

    // 横向、纵向分块数量
    const int nCols = width / W;   // 横向分块数量，分块列数
    const int nRows = height / W;  // 纵向分块数量，分块行数
    // 每个网格块的宽和高   >= 30
    const int wCell = ceil(width / nCols);   // 网格块宽
    const int hCell = ceil(height / nRows);  // 网格块高

    // 遍历每一行网格
    for (int i = 0; i < nRows; i++) {
      const float iniY = minBorderY + i * hCell;  // 原点 y 坐标
      float maxY = iniY + hCell + 6;              // 下方 y 坐标

      if (iniY >= maxBorderY - 3) continue;
      if (maxY > maxBorderY) maxY = maxBorderY;

      // 遍历行中的每一列(个)网格
      for (int j = 0; j < nCols; j++) {
        const float iniX = minBorderX + j * wCell;  // 原点 x 坐标
        float maxX = iniX + wCell + 6;              // 右侧 x 坐标
        if (iniX >= maxBorderX - 6) continue;
        if (maxX > maxBorderX) maxX = maxBorderX;

        // 网格中的特征点
        vector<cv::KeyPoint> vKeysCell;
        // FAST 特征提取
        // (图像, v特征点, 阈值, 是否非极大值抑制)
        FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
             vKeysCell, iniThFAST, true);

        // 如果没有提取到特征点，用小阈值提取
        if (vKeysCell.empty()) {
          FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
               vKeysCell, minThFAST, true);
        }

        // 如果提取到了特征点，则加入到待分配特征点集合
        if (!vKeysCell.empty()) {
          for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin();
               vit != vKeysCell.end(); vit++) {
            (*vit).pt.x += j * wCell;  // 金字塔图像中的坐标
            (*vit).pt.y += i * hCell;
            vToDistributeKeys.push_back(*vit);
          }
        }
      }
    }

    // 这一层的特征点
    vector<KeyPoint>& keypoints = allKeypoints[level];
    keypoints.reserve(nfeatures);

    // 分配特征点
    keypoints =
        DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY,
                          maxBorderY, mnFeaturesPerLevel[level], level);

    const int scaledPatchSize =
        PATCH_SIZE * mvScaleFactor[level];  // 特征点尺度

    // Add border to coordinates and scale information
    const int nkps = keypoints.size();
    for (int i = 0; i < nkps; i++) {
      keypoints[i].pt.x += minBorderX;      // 金字塔图像中的 x 坐标
      keypoints[i].pt.y += minBorderY;      // 金字塔图像中的 y 坐标
      keypoints[i].octave = level;          // 出自金字塔层数
      keypoints[i].size = scaledPatchSize;  // 特征尺度
    }
  }

  // compute orientations
  // 计算特征方向
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

/// @brief 提取特征点  网格划分法
/// @param allKeypoints 每层金字塔的特征点集合
void ORBextractor::ComputeKeyPointsOld(
    std::vector<std::vector<KeyPoint> >& allKeypoints) {
  allKeypoints.resize(nlevels);

  float imageRatio =
      (float)mvImagePyramid[0].cols / mvImagePyramid[0].rows;  // 图像宽高比

  // 遍历金字塔每一层，划分网格，在每个网格中提取角点
  for (int level = 0; level < nlevels; ++level) {
    const int nDesiredFeatures = mnFeaturesPerLevel[level];  // 每层特征数

    // 横向、纵向分块数量  自适应方法
    const int levelCols = sqrt((float)nDesiredFeatures /
                               (5 * imageRatio));  // 横向分块数量，分块列数
    const int levelRows = imageRatio * levelCols;  // 纵向分块数量，分块行数

    // 角点提取边界
    const int minBorderX = EDGE_THRESHOLD;  // 左上
    const int minBorderY = minBorderX;
    const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;  // 右下
    const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

    // 提取区域宽、高
    const int W = maxBorderX - minBorderX;
    const int H = maxBorderY - minBorderY;
    // 每个网格块的宽和高
    const int cellW = ceil((float)W / levelCols);  // 网格块宽
    const int cellH = ceil((float)H / levelRows);  // 网格块高

    const int nCells = levelRows * levelCols;  // 网格总数
    const int nfeaturesCell =
        ceil((float)nDesiredFeatures / nCells);  // 每个网格特征数，均匀分配

    // 特征点   维度: 网格行 网格列 网格中特征点集合 特征点
    vector<vector<vector<KeyPoint> > > cellKeyPoints(
        levelRows, vector<vector<KeyPoint> >(levelCols));

    // 维度: 网格行 网格列 网格中需要保留的特征点数目
    vector<vector<int> > nToRetain(
        levelRows, vector<int>(levelCols, 0));  // 网格中需要保留的特征点数目
    // 维度: 网格行 网格列 网格中特征点总数
    vector<vector<int> > nTotal(
        levelRows, vector<int>(levelCols, 0));  // 网格中提取到的特征点总数
    // 维度: 网格行 网格列 是否小于
    vector<vector<bool> > bNoMore(
        levelRows,
        vector<bool>(levelCols,
                     false));  // 网格中特征点总数是否小于目标提取总数
    vector<int> iniXCol(levelCols);  // 网格块左上角列位置, x 坐标
    vector<int> iniYRow(levelRows);  // 网格块左上角行位置, y 坐标
    int nNoMore = 0;  // 特征点数小于目标提取总数的网格总数
    int nToDistribute = 0;  // 需要从其他网格中分配的特征数

    float hY = cellH + 6;  // 网格高，多 6 个像素   为啥呢？

    // 遍历每一行网格
    for (int i = 0; i < levelRows; i++) {
      const float iniY = minBorderY + i * cellH - 3;  // 原点 y 坐标
      iniYRow[i] = iniY;

      // 最后一行加判断
      if (i == levelRows - 1) {
        hY = maxBorderY + 3 - iniY;
        if (hY <= 0) continue;
      }

      float hX = cellW + 6;  // 网格宽

      // 遍历行中的每一列(个)网格
      for (int j = 0; j < levelCols; j++) {
        float iniX;

        // 只对第一行计算就可以，其余行都相同
        if (i == 0) {
          iniX = minBorderX + j * cellW - 3;
          iniXCol[j] = iniX;
        } else {
          iniX = iniXCol[j];
        }

        // 最后一列加判断
        if (j == levelCols - 1) {
          hX = maxBorderX + 3 - iniX;
          if (hX <= 0) continue;
        }

        // 网格图像
        Mat cellImage = mvImagePyramid[level]
                            .rowRange(iniY, iniY + hY)
                            .colRange(iniX, iniX + hX);

        // 预先分配五倍空间
        cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

        // 提取 FAST 特征
        FAST(cellImage, cellKeyPoints[i][j], iniThFAST, true);

        // 如果提取到的特征点数小于 3，用小阈值提取
        if (cellKeyPoints[i][j].size() <= 3) {
          cellKeyPoints[i][j].clear();

          FAST(cellImage, cellKeyPoints[i][j], minThFAST, true);
        }

        const int nKeys = cellKeyPoints[i][j].size();
        nTotal[i][j] = nKeys;

        if (nKeys > nfeaturesCell) {
          // 多于目标数
          nToRetain[i][j] = nfeaturesCell;
          bNoMore[i][j] = false;
        } else {
          // 小于目标数，需要从其他网格中匀
          nToRetain[i][j] = nKeys;
          nToDistribute += nfeaturesCell - nKeys;
          bNoMore[i][j] = true;
          nNoMore++;
        }
      }
    }

    // Retain by score
    // 特征点分配
    // 分配剩余特征点给有多余特征点的网格
    // 如果还有需要分配的特征点，且还有网格有多余特征点
    while (nToDistribute > 0 && nNoMore < nCells) {
      // 还有多余特征点的网格均匀分担
      int nNewFeaturesCell =
          nfeaturesCell + ceil((float)nToDistribute / (nCells - nNoMore));
      nToDistribute = 0;

      // 遍历行
      for (int i = 0; i < levelRows; i++) {
        // 遍历列
        for (int j = 0; j < levelCols; j++) {
          // 如果有多余特征点
          if (!bNoMore[i][j]) {
            if (nTotal[i][j] > nNewFeaturesCell) {
              // 如果分完还有多余
              nToRetain[i][j] = nNewFeaturesCell;
              bNoMore[i][j] = false;
            } else {
              // 剩余特征点不够分配
              nToRetain[i][j] = nTotal[i][j];
              nToDistribute += nNewFeaturesCell - nTotal[i][j];
              bNoMore[i][j] = true;
              nNoMore++;
            }
          }
        }
      }
    }

    vector<KeyPoint>& keypoints = allKeypoints[level];  // 当前层特征点集合
    keypoints.reserve(nDesiredFeatures * 2);

    const int scaledPatchSize =
        PATCH_SIZE * mvScaleFactor[level];  // 特征点尺度

    // Retain by score and transform coordinates
    // 特征点筛选
    for (int i = 0; i < levelRows; i++) {
      for (int j = 0; j < levelCols; j++) {
        vector<KeyPoint>& keysCell = cellKeyPoints[i][j];
        // 根据特征响应(分数)筛选指定数量特征点
        KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
        if ((int)keysCell.size() > nToRetain[i][j])
          keysCell.resize(nToRetain[i][j]);

        for (size_t k = 0, kend = keysCell.size(); k < kend; k++) {
          keysCell[k].pt.x += iniXCol[j];  // 金字塔图像中的 x 坐标
          keysCell[k].pt.y += iniYRow[i];  // 金字塔图像中的 y 坐标
          // keysCell[k].response  // 特征响应，即得分
          keysCell[k].octave = level;          // 出自金字塔层数
          keysCell[k].size = scaledPatchSize;  // 特征尺度
          keypoints.push_back(keysCell[k]);
        }
      }
    }

    // 再筛选一次，保证不超过总数   为啥会超过呢...
    if ((int)keypoints.size() > nDesiredFeatures) {
      KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
      keypoints.resize(nDesiredFeatures);
    }
  }

  // and compute orientations
  // 计算特征方向
  for (int level = 0; level < nlevels; ++level)
    computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

/// @brief 计算 BRIEF 描述子
/// @param image 图像
/// @param keypoints 特征点
/// @param descriptors 描述子
/// @param pattern 匹配模板
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints,
                               Mat& descriptors, const vector<Point>& pattern) {
  descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

  // 每个点单算
  for (size_t i = 0; i < keypoints.size(); i++)
    computeOrbDescriptor(keypoints[i], image, &pattern[0],
                         descriptors.ptr((int)i));
}

/// @brief 提取 ORB 特征，计算描述子    运算符重载
/// @param _image 图像
/// @param _mask 掩码，弃用
/// @param _keypoints 关键点
/// @param _descriptors 描述子
void ORBextractor::operator()(InputArray _image, InputArray _mask,
                              vector<KeyPoint>& _keypoints,
                              OutputArray _descriptors) {
  if (_image.empty()) return;

  Mat image = _image.getMat();
  assert(image.type() == CV_8UC1);

  // Pre-compute the scale pyramid
  ComputePyramid(image);

  vector<vector<KeyPoint> > allKeypoints;  // 每层的特征点集合
  ComputeKeyPointsOctTree(allKeypoints);
  // ComputeKeyPointsOld(allKeypoints);

  Mat descriptors;

  int nkeypoints = 0;  // 特征点总数
  for (int level = 0; level < nlevels; ++level)
    nkeypoints += (int)allKeypoints[level].size();
  if (nkeypoints == 0)
    _descriptors.release();
  else {
    // nkeypoints 个特征点   32 * 8 = 256 位描述子
    _descriptors.create(nkeypoints, 32, CV_8U);
    // getMat 是获取指针吗 ?
    descriptors = _descriptors.getMat();
  }

  _keypoints.clear();
  _keypoints.reserve(nkeypoints);

  int offset = 0;  // 当前层特征点序号

  // 逐层计算描述子
  for (int level = 0; level < nlevels; ++level) {
    vector<KeyPoint>& keypoints = allKeypoints[level];
    int nkeypointsLevel = (int)keypoints.size();

    if (nkeypointsLevel == 0) continue;

    // preprocess the resized image
    Mat workingMat = mvImagePyramid[level].clone();
    // 高斯平滑
    GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    // Compute the descriptors
    Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    computeDescriptors(workingMat, keypoints, desc, pattern);

    offset += nkeypointsLevel;

    // Scale keypoint coordinates
    if (level != 0) {
      // 小图上获得的是尺寸更大的特征点  scale > 1
      float scale =
          mvScaleFactor[level];  // getScale(level, firstLevel, scaleFactor);

      // 注意!!  这里将特征坐标恢复到了金字塔底层，即绝对坐标
      for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                      keypointEnd = keypoints.end();
           keypoint != keypointEnd; ++keypoint)
        keypoint->pt *= scale;
    }
    // And add the keypoints to the output
    _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
  }
}

/// @brief 计算图像金字塔
/// @param image
void ORBextractor::ComputePyramid(cv::Mat image) {
  for (int level = 0; level < nlevels; ++level) {
    float scale = mvInvScaleFactor[level];  // 逆尺寸，小于 1
    Size sz(cvRound((float)image.cols * scale),
            cvRound((float)image.rows * scale));
    Size wholeSize(sz.width + EDGE_THRESHOLD * 2,
                   sz.height + EDGE_THRESHOLD * 2);  // 预留 BRIEF 计算边界
    // 为什么需要这两个矩阵 ?    temp 是扩充边界的图像
    // 这些部分不能省略
    Mat temp(wholeSize, image.type()), masktemp;
    // 这里应该是浅拷贝
    mvImagePyramid[level] =
        temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

    // Compute the resized image
    if (level != 0) {
      resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0,
             INTER_LINEAR);

      // 创建边界
      copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD,
                     EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     BORDER_REFLECT_101 + BORDER_ISOLATED);
    } else {
      copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                     EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101);
    }
  }
}

}  // namespace ORB_SLAM2
