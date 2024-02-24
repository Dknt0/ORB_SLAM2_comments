/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"  // 块求解器
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"  // 列马法
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"  // 线性求解器用 Eigen
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"  // SE(3) 指数映射
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"  // 核函数
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"  // 稠密求解器
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"  // Sim(3) 指数映射

#include <Eigen/StdVector>

#include "Converter.h"

#include <mutex>

namespace ORB_SLAM2
{

/// @brief 全局光束法平差
///     LoopClosing::RunGlobalBundleAdjustment
/// @param pMap 地图
/// @param nIterations 迭代次数
/// @param pbStopFlag 停止标志位
/// @param nLoopKF 回环关键帧 id
/// @param bRobust 是否使用核函数
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}

/// @brief 光束法平差
/// @param vpKFs 优化关键帧
/// @param vpMP 优化地图点
/// @param nIterations 迭代次数
/// @param pbStopFlag 停止标志位
/// @param nLoopKF 回环关键帧 id
/// @param bRobust 是否使用核函数
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;  // 优化不包含节点标志位
    vbNotIncludedMP.resize(vpMP.size());

    /* 1.配置求解器 */
    g2o::SparseOptimizer optimizer;  // 图模型
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;  // 线性求解器  6*3
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);  // 块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);  // 图优化求解器
    optimizer.setAlgorithm(solver);
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);  // 设置强制停止标志

    /* 2.遍历优化 KF 集，添加 KF 顶点 */
    long unsigned int maxKFid = 0;  // 最大关键帧 id

    // Set KeyFrame vertices
    // 遍历关键帧集，设置关键帧图优化节点
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];  // 优化关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();  // g2o SE3 类
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));  // 初值
        vSE3->setId(pKF->mnId);  // id 同关键帧序号
        vSE3->setFixed(pKF->mnId==0);  // 第 0 帧为固定帧
        optimizer.addVertex(vSE3);  // 添加节点到图
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;  // 记录最大关键帧序号
    }

    // 卡方分布 95% 以上可信度的时候的阈值  用于设置核函数、外点筛选
    const float thHuber2D = sqrt(5.99);  // Huber 核函数阈值
    const float thHuber3D = sqrt(7.815);  // Huber 核函数阈值

    /* 3.遍历优化 MP 集，添加 MP 顶点，按照 MP 与对应 KF 的观测关系添加边 */

    // Set MapPoint vertices
    // 遍历节点集，设置地图点图优化节点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];  // 优化地图点
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();  // g2o v3d 类
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));  // 初值
        const int id = pMP->mnId+maxKFid+1;  // 保证节点 id 不与关键帧节点 id 重合
        vPoint->setId(id);
        vPoint->setMarginalized(true);  // 边缘化，什么意思?
        optimizer.addVertex(vPoint);  // 添加节点到图

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();  // 观测关键帧集

        int nEdges = 0;  // 边数
        // SET EDGES
        // 遍历地图点的观测集，设置图优化边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {
            KeyFrame* pKF = mit->first;  // 关键帧
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];  // 去畸变关键点

            // 如果深度无效
            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;  // 二维观测坐标
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();  // 边 空间点到成像平面

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  // 观测地图点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));  // 观测关键帧
                e->setMeasurement(obs);  // 设置观测结果
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);  // 信息矩阵  层数越低越大
                
                // 使用鲁棒核函数
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);  // 设置鲁棒核函数
                    rk->setDelta(thHuber2D);
                }

                // 相机内参
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);  // 添加边到图
            }
            // 如果深度有效
            else
            {
                Eigen::Matrix<double,3,1> obs;  // 三维观测坐标
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();  // 边 空间点到双目成像平面

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  // 设置地图点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));  // 设置关键帧
                e->setMeasurement(obs);  // 设置观测结果
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;  // 信息矩阵  层数越低越大，与深度无关
                e->setInformation(Info);

                // 使用鲁棒核函数
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);  // 设置鲁棒核函数
                    rk->setDelta(thHuber3D);
                }

                // 相机内参
                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);  // 添加边到图
            }
        }

        // 如果地图点没有有效观测信息，忽略
        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;  // 设置标志位
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    /* 4.优化 */
    // Optimize!
    // 优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    /* 5.从优化结果恢复 KF, MP 数据 */
    // Recover optimized data

    // Keyframes
    // 恢复关键帧数据
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];  // 关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));  // 图模型关键帧节点
        g2o::SE3Quat SE3quat = vSE3->estimate();  // 位姿类型
        if(nLoopKF==0)
        {
            // 回环帧为 0  在单目初始化中用到
            pKF->SetPose(Converter::toCvMat(SE3quat));  // 位姿
        }
        else
        {
            // 其他
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);  // 暂存位姿
            pKF->mnBAGlobalForKF = nLoopKF;  // 暂存回环帧 id
        }
    }

    // Points
    // 恢复地图点数据
    for(size_t i=0; i<vpMP.size(); i++)
    {
        // 优化不包含此地图点，跳过
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];  // 地图点

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));  // 图模型地图点节点

        if(nLoopKF==0)
        {
            // 回环帧为 0  在单目初始化中用到
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();  // 更新地图点平均观测方向和尺度无关距离
        }
        else
        {
            // 其他
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);  // 暂存位置
            pMP->mnBAGlobalForKF = nLoopKF;  // 暂存回环帧 id
        }
    }

}

/// @brief 单帧估计 记录外点信息  Motion-only BA
///     Tracking::TrackReferenceKeyFrame
///     Tracking::TrackWithMotionModel
///     Tracking::Relocalization
/// @param pFrame 帧
/// @return 内点数量
int Optimizer::PoseOptimization(Frame *pFrame)
{
    /* 1.配置求解器 */

    g2o::SparseOptimizer optimizer;  // 图模型
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;  // 线性求解器
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);  // 块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    /* 2.添加 KF 顶点 */

    /* 设置顶点和边 */
    // Set Frame vertex
    // 设置帧顶点
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));  // 初始位姿
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);  // 添加顶点到图
    
    /* 3.依据 KF 地图点观测添加单目或双目观测边 */

    int nInitialCorrespondences=0;  // 初始地图点观测数量
    // Set MapPoint vertices
    const int N = pFrame->N; // 左图关键点数量

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;  // 单目观测边集
    vector<size_t> vnIndexEdgeMono;  // 单目观测关键点序号集
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;  // 双目观测边集
    vector<size_t> vnIndexEdgeStereo;  // 双目观测关键点序号集
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    // 卡方分布 95% 以上可信度的时候的阈值
    const float deltaMono = sqrt(5.991);  // Huber 核函数阈值
    const float deltaStereo = sqrt(7.815);  // Huber 核函数阈值


    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);  // 地图点全局锁

        // 遍历所有关键点
        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];  // 地图点
            // 如果存在观测关系
            if(pMP)
            {
                // Monocular observation
                // 如果是单目点
                if(pFrame->mvuRight[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;  // 存在地图点匹配，不是外点

                    Eigen::Matrix<double,2,1> obs;  // 二维观测
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];  // 关键点
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();  // 边 空间点到平面 Motion-only

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  // 观测关键帧
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);  // 信息矩阵  层数越低越大

                    // 鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    // 设置相机内参
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    // 设置地图点世界坐标
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);  // 添加边到图

                    vpEdgesMono.push_back(e);  // 记录边
                    vnIndexEdgeMono.push_back(i);  // 记录关键点
                }
                // 如果是双目点
                else  // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;  // 存在地图点匹配，不是外点

                    //SET EDGE
                    Eigen::Matrix<double,3,1> obs;  // 三维观测
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];  // 关键点
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 边 空间点到双目平面 Motion-only
                    e->setMeasurement(obs);  // 设置观测结果
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;  // 信息矩阵  层数越低越大
                    e->setInformation(Info);

                    // 鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    // 设置相机内参
                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    // 设置地图点世界坐标
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);  // 添加边到图

                    vpEdgesStereo.push_back(e);  // 记录边
                    vnIndexEdgeStereo.push_back(i);  // 记录关键点
                }
            }

        }
    }

    // 小于 3 个地图点观测时问题无解
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 进行四次优化，每次优化后会重新将观测分为内点和外点，在下一测观测中不会考虑外点，但最终他们可能被再次分类为内点
    
    // 核函数阈值
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    // 迭代次数
    const int its[4]={10,10,10,10};    

    /* 4.四步优化，外点去除 */

    int nBad=0;  // 坏点数量
    // 进行四次优化
    // 这里第四次优化后也进行了外电剔除，实际上没有意义
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));  // 设置定点初值
        // 优化
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;  // 重新计算外点
        // 遍历单目观测边
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];  // 单目观测边

            const size_t idx = vnIndexEdgeMono[i];  // 单目关键点序号
            
            // 若上一次优化中被设置为外点，则重新计算误差
            // 之前被当作外点的点，也可能重新作为内点使用
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }
            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                // 误差大于阈值，设置为外点
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);  // 不优化
                nBad++;
            }
            else
            {
                // 误差小于阈值，设置为内点
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            // 最后一轮迭代不使用核函数
            if(it==2)
                e->setRobustKernel(0);
        }

        // 遍历双目观测边
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];  // 双目观测边

            const size_t idx = vnIndexEdgeStereo[i];  // 双目关键点序号

            // 若上一次优化中被设置为外点，则重新计算误差
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }
            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                // 误差大于阈值，设置为外点
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);  // 不优化
                nBad++;
            }
            else
            {
                // 误差小于阈值，设置为内点
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            // 最后一轮迭代不使用核函数
            if(it==2)
                e->setRobustKernel(0);
        }

        // 若剩余观测数小于 10 结束
        if(optimizer.edges().size()<10)
            break;
    }

    /* 5.恢复 KF 位姿 */

    // Recover optimized pose and return number of inliers
    // 恢复位姿
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

/// @brief 局部 BA
/// @param pKF 关键帧
/// @param pbStopFlag 停止标志位
/// @param pMap 地图
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    /* 1.从输入 KF 共视图获取局部 KF */

    // Local KeyFrames: First Breath Search from Current Keyframe
    // 与优化关键帧共视数多于阈值的关键帧，为局部关键帧
    list<KeyFrame*> lLocalKeyFrames;  // 局部关键帧列表

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;  // 设置优化帧为 pKF

    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();  // 有序共视关键帧向量
    // 遍历共视帧向量，只要不为坏关键帧都参与优化
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];  // 共视帧
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    /* 2.从局部 KF 观测关系获取局部 MP */

    // Local MapPoints seen in Local KeyFrames
    // 局部关键帧观测到的地图点为局部地图点
    list<MapPoint*> lLocalMapPoints;  // 局部地图点列表
    // 遍历局部关键帧列表
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();  // 获取关键点对应的地图点观测
        // 遍历关键点
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            // 如果存在观测，且不为坏点
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        // 如果地图点没有被添加到这次局部 BA 中，则添加
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    /* 3.遍历局部 MP 观测关系，确定固定 KF。观测到局部 MP，且不是局部 KF 的，为固定 KF */

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 观测到局部地图点，但不是局部关键帧的，为固定关键帧
    list<KeyFrame*> lFixedCameras;  // 固定关键帧列表
    // 遍历局部地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();  // 观测到地图点的关键帧
        // 遍历观测关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;  // 关键帧
            // 如果关键帧不是局部关键帧，且没有被添加到这次局部 BA 的固定关键帧中，则添加
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    /* 4.求解器配置 */

    // Setup optimizer
    g2o::SparseOptimizer optimizer;  // 图模型
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;  // 线性求解器
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);  // 块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);  // 图优化求解器
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);  // 停止标志位置

    /* 5.添加局部 KF 顶点 */

    unsigned long maxKFid = 0;  // 最大关键帧序号，用于设置地图点节点序号

    // Set Local KeyFrame vertices
    // 设置局部关键帧节点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));  // 初值
        vSE3->setId(pKFi->mnId);  // 序号同关键帧序号
        vSE3->setFixed(pKFi->mnId==0);  // 仅初始帧设置为固定
        optimizer.addVertex(vSE3);  // 添加顶点
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;  // 记录最大关键帧序号
    }

    /* 6.添加固定 KF 顶点，开启固定 */

    // Set Fixed KeyFrame vertices
    // 设置固定关键帧节点
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);  // 固定
        optimizer.addVertex(vSE3);  // 添加顶点
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;  // 记录最大关键帧序号
    }

    /* 7.添加 MP 顶点，开启边缘化  添加单目、双目边 */

    // Set MapPoint vertices
    // 设置地图点节点
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();  // 最大观测数

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;  // 单目观测边
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;  // 单目观测关键帧
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;  // 单目观测地图点
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;  // 双目观测边
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;  // 双目观测关键帧
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;  // 双目观测地图点
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    // 卡方分布 95% 以上可信度的时候的阈值
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // 遍历局部地图点，确定观测边
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;  // 地图点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();  // 地图点节点
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));  // 初值
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);  // 边缘化
        optimizer.addVertex(vPoint);  // 添加节点到图

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();  // 地图点观测集

        //Set edges
        // 遍历局部地图点观测集，确定观测边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;  // 关键帧

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];  // 关键点

                // Monocular observation
                // 视差无效，单目观测
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;  // 二维观测
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();  // 边 空间点到成像平面

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  // 观测地图点
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));  // 观测关键帧
                    e->setMeasurement(obs);  // 设置观测结果
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);  // 信息矩阵  层数越低越大

                    // 使用鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    // 相机内参
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);  // 添加边到图
                    // 记录单目观测到向量
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // 视差有效，双目观测
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;  // 三维观测
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();  // 边 空间点到双目成像平面

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  // 设置地图点
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));  // 设置关键帧
                    e->setMeasurement(obs);  // 设置观测结果
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];  // 逆面积比例 <1
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;  // 信息矩阵  层数越低越大，与深度无关
                    e->setInformation(Info);
                    
                    // 使用鲁棒核函数
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    // 相机内参
                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);  // 添加边到图
                    // 记录双目观测到向量
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    /* 8.初步优化 */

    // 第一步优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    /* 9.计算残差，剔除外点 */

    if(bDoMore)
    {

        // Check inlier observations
        // 遍历单目观测，检查内点
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];  // 观测边
            MapPoint* pMP = vpMapPointEdgeMono[i];  // 观测地图点

            if(pMP->isBad())
                continue;
            
            // 如果误差超过阈值，或深度为负，设置为外点
            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);  // 固定
            }

            e->setRobustKernel(0);  // 第二步优化不使用核函数
        }

        // 遍历双目观测，检查内点
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];  // 观测边
            MapPoint* pMP = vpMapPointEdgeStereo[i];  // 观测地图点

            if(pMP->isBad())
                continue;
            
            // 如果误差超过阈值，或深度为负，设置为外点
            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);  // 固定
            }

            e->setRobustKernel(0);  // 第二步优化不使用核函数
        }

        /* 10.再次优化 */

        // Optimize again without the outliers
        // 第二步优化
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;  // 待清除观测集
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());
    
    /* 11.外点检查，清除外点边对应的 KF, MP 间的观测关系 */

    // Check inlier observations
    // 遍历单目观测边，检查内点
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];  // 边
        MapPoint* pMP = vpMapPointEdgeMono[i];  // 地图点

        if(pMP->isBad())
            continue;

        // 如果误差超过阈值，或深度为负，记录为外点
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // 遍历双目观测边，检查内点
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        // 如果误差超过阈值，或深度为负，记录为外点
        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);  // 地图更新锁

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);  // 清除关键帧对地图点的观测
            pMPi->EraseObservation(pKFi);  // 清除地图点对关键帧的观测
        }
    }

    /* 12.从优化结果恢复 KF, MP 结果 */

    // Recover optimized data
    // 恢复数据

    //Keyframes
    // 恢复局部关键帧数据
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));  // 设置位姿
    }

    //Points
    // 恢复局部地图点数据
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));  // 设置位置
        pMP->UpdateNormalAndDepth();  // 更新地图点平均观测方向和尺度无关距离
    }
}

/// @brief 本质图优化  LoopClosing 中调用
/// @param pMap 地图
/// @param pLoopKF 回环匹配关键帧  与 pLoopKF 匹配
/// @param pCurKF 当前关键帧
/// @param NonCorrectedSim3 未修正 Sim3 映射<关键帧, Sim3>
/// @param CorrectedSim3 已修正 Sim3 映射<关键帧, Sim3>
/// @param LoopConnections 回环连接集 映射<关键帧, 集合<回环关键帧>>
/// @param bFixScale 是否修正尺度
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    /* 1.配置求解器 */
    g2o::SparseOptimizer optimizer;  // 图模型
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();  // 线性求解器
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);  // 块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);  // 图优化求解器

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();  // 全部关键帧
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();  // 全部地图点

    const unsigned int nMaxKFid = pMap->GetMaxKFid();  // 最大关键帧序号

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);  // 优化前位姿 Scw
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);  // 优化后位姿 Scw
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);  // 图模型节点

    const int minFeat = 100;  // 最小关键帧阈值

    /* 2.添加 KF 节点 */

    // Set KeyFrame vertices
    // 遍历全部关键帧，添加节点
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];  // 关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();  // 本质图节点

        const int nIDi = pKF->mnId;  // 关键帧序号

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);  // 在修正位姿映射中查询当前关键帧

        if(it!=CorrectedSim3.end())
        {
            // 如果存在修正结果，使用之
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            // 如果不存在修正结果，使用关键帧当前位姿
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        // 固定回环关键帧  注意，第 0 帧不固定
        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);  // 节点序号同关键帧序号
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;  // 是否固定尺度

        optimizer.addVertex(VSim3);  // 添加节点到图

        vpVertices[nIDi]=VSim3;  // 记录节点
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;  // 回环边两端的节点 <关键帧序号, 关键帧序号>

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();  // Lambda 矩阵, 作为信息矩阵

    /* 3.添加当前回环边 */

    // Set Loop edges
    // 遍历回环连接集，设置回环边
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;  // 关键帧
        const long unsigned int nIDi = pKF->mnId;  // 关键帧 id
        const set<KeyFrame*> &spConnections = mit->second;  // 连接关键帧集合
        const g2o::Sim3 Siw = vScw[nIDi];  // 修正前 Siw
        const g2o::Sim3 Swi = Siw.inverse();  // Swi

        // 遍历连接关键帧集合
        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;  // 连接关键帧 id
            // 如果非当前关键帧、非回环关键帧 且 共视数小于阈值 忽略
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];  // 修正前 Sjw
            const g2o::Sim3 Sji = Sjw * Swi;  // Sji

            g2o::EdgeSim3* e = new g2o::EdgeSim3();  // 回环边节点
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;  // 信息矩阵

            optimizer.addEdge(e);  // 添加边到因子图

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));  // 记录边的连接关系
        }
    }

    /* 4.添加本质图边，包括生成树、历史回环边、共视图边 */

    // Set normal edges
    // 遍历全部关键帧，添加本质图边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];  // 关键帧

        const int nIDi = pKF->mnId;  // 关键帧 id

        g2o::Sim3 Swi;  // Swi

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);  // 在未修正位姿映射中查询当前关键帧

        if(iti!=NonCorrectedSim3.end())
            // 如果存在结果，则使用
            Swi = (iti->second).inverse();
        else
            // 不存在结果，使用 vScw 中的位姿
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();  // 父关键帧

        // Spanning tree edge
        // 生成树边   只连接当前节点和其父节点
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;  // 父关键帧 id

            g2o::Sim3 Sjw;  // Sjw

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);  // 在未修正位姿映射中查询父关键帧

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;  // Sji

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            // 这些 dynamic_cast 有意义吗?
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 回环边   当前节点的回环边  这里是过去检测出的回环边，不包括本次回环
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();  // 回环边集合
        // 遍历回环边
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;  // 回环关键帧
            // 保证不重复添加回环边
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);  // 在未修正位姿映射中查询父关键帧

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        // 共视图边
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);  // 超过阈值的共视图边
        // 遍历共视图边
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;  // 共视关键帧
            // pKFn有效 且 pKFn不是父关键帧 且 pKFn不是子关键帧 且 pKFn不构成回环边   即，pKFn 与当前帧的共视关系不属于生成树与回环边
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                // 保证不重复添加共视图边
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    // 保证不是当前回环边，这些边已经考虑过了
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);// 在未修正位姿映射中查询父关键帧

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    /* 5.优化 */

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);  // 地图锁

    /* 6.依据结果恢复 KF 位姿 */

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 遍历关键帧，修正关键帧位姿
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];  // 关键帧

        const int nIDi = pKFi->mnId;  // 关键帧 id

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));  // 节点
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();  // 位姿
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();  // 记录优化后位姿
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]  // 尺度修正

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    /* 7.修正地图点 */

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    // 修正地图点，先变换到优化前的关键帧系下，再按照优化后的关键帧位姿变换回世界
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];  // 地图点

        if(pMP->isBad())
            continue;

        int nIDr;  // 参考关键帧 id
        // 如果关键帧被当前关键帧修正
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        // 其他
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();  // 参考关键帧
            nIDr = pRefKF->mnId;
        }

        g2o::Sim3 Srw = vScw[nIDr];  // 优化前关键帧位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];  // 优化后关键帧位姿

        cv::Mat P3Dw = pMP->GetWorldPos();  // 地图点坐标
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));  // 修正后世界坐标

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();  // 更新地图点平均观测方向和尺度无关距离
    }
}

/// @brief Sim3 优化  检查候选关键帧  1投2，2投1  ICP 问题迭代解
///     LoopClosing::ComputeSim3
/// @param pKF1 当前关键帧
/// @param pKF2 候选关键帧
/// @param vpMatches1 KF1 对 KF2 MP 匹配  按照 KF1 KP 索引
/// @param g2oS12 相似变换  2 相对 1
/// @param th2 误差上限阈值的平方
/// @param bFixScale 是否固定尺度
/// @return 内点数量
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    /* 1.求解器配置 */
    g2o::SparseOptimizer optimizer;  // 图模型
    g2o::BlockSolverX::LinearSolverType * linearSolver;  // 线性求解器
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);  // 块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);  // 图优化求解器
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;  // 帧 1 内参
    const cv::Mat &K2 = pKF2->mK;  // 帧 2 内参

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();  // 帧 1 姿态
    const cv::Mat t1w = pKF1->GetTranslation();  // 帧 1 位置
    const cv::Mat R2w = pKF2->GetRotation();  // 帧 2 姿态
    const cv::Mat t2w = pKF2->GetTranslation();  // 帧 2 位置

    /* 2.添加帧间变换 Sim3 顶点 */

    // Set Sim3 vertex
    // 设置 Sim3 节点
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap(); // 节点
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);  // 初值
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);  // cx1
    vSim3->_principle_point1[1] = K1.at<float>(1,2);  // cy1
    vSim3->_focal_length1[0] = K1.at<float>(0,0);  // fx1
    vSim3->_focal_length1[1] = K1.at<float>(1,1);  // fy1
    vSim3->_principle_point2[0] = K2.at<float>(0,2);  // cx2
    vSim3->_principle_point2[1] = K2.at<float>(1,2);  // cy2
    vSim3->_focal_length2[0] = K2.at<float>(0,0);  // fx2
    vSim3->_focal_length2[1] = K2.at<float>(1,1);  // fy2
    optimizer.addVertex(vSim3);  // 添加节点到图

    /* 3.添加 MP 顶点，按照观测关系添加边 */

    // Set MapPoint vertices
    // 设置地图点节点
    const int N = vpMatches1.size();  // 关键点数
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();  // 关键帧 1 地图点匹配
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;  // Sim3 投影边
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;  // Sim3 逆投影边
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);  // 阈值

    int nCorrespondences = 0;  // 观测数

    // 遍历当前关键帧地图点匹配关键点
    for(int i=0; i<N; i++)
    {
        // 如果关键点不存在地图点匹配，忽略
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];  // KP1  来自 KF1 MP 匹配
        MapPoint* pMP2 = vpMatches1[i];  // KP2  KF1 对 KF2 MP 观测

        const int id1 = 2*i+1;  // 关键点 1 序号
        const int id2 = 2*(i+1);  // 关键点 2 序号

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);  // pMP2 在 pKF2 中的关键点序号

        // 如果 pMP1 和 pMP2 都有效
        if(pMP1 && pMP2)
        {
            // 如果 pMP1 和 pMP2 不为坏点，且存在 pKF2 对 pMP2 的观测
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();  // MP1 节点
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;  // MP1 在 KF1 相机系坐标
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();  // MP1 节点
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;  // MP2 在 KF2 相机系坐标
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        // 帧1 对 点2
        Eigen::Matrix<double,2,1> obs1;  // 二维观测
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;
        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();  // 边  Sim3 投影
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));  // 点2
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  // Sim12
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);  // 信息矩阵与层数相关
        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;  // 核函数
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        // 帧2 对 点1
        Eigen::Matrix<double,2,1> obs2;  // 二维观测
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;
        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();  // 边  Sim3 逆投影
        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));  // 点1
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));  // Sim12
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);  // 信息矩阵与层数相关
        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;  // 核函数
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);  // 保存投影边
        vpEdges21.push_back(e21);  // 保存逆投影边
        vnIndexEdge.push_back(i);  // 边序号
    }

    /* 4.初步优化 5 轮 */

    // Optimize!
    // 初步优化
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    /* 5.计算重投影误差，剔除内点 */

    // Check inliers
    // 检查内点
    int nBad=0;  // 外点数
    // 遍历所有边，筛选内点
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];  // 获取 Sim 投影边
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];  // 获取 Sim 逆投影边
        if(!e12 || !e21)
            continue;

        // 只要有一个边的残差大于阈值，就认为这一对观测是外点
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);  // 匹配记为失败
            optimizer.removeEdge(e12);  // 删除边
            optimizer.removeEdge(e21);  // 删除边
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }
    
    int nMoreIterations;  // 再次优化迭代次数
    // 如果存在外点，优化 10 次
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    /* 6.再次优化 10 轮 */

    // Optimize again only with inliers
    // 再次优化
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    /* 7.计算重投影误差，清除 KF 对 MP 的观测 */

    int nIn = 0;  // 内点数
    // 遍历所有边，筛选内点
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        // 只要有一个边的残差大于阈值，就认为这一对观测是外点
        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    /* 8.从优化结果获取帧间变换结果 */

    // Recover optimized Sim3
    // 恢复数据
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;  // 返回内点数
}


} //namespace ORB_SLAM
