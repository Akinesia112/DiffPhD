#pragma once
#include "Scomponent/collisiondetection/BaseCollision.h"
#include "Sgeometry/SBaseGeometry.h"

#include "tbb/tbb.h"

namespace RealSim::collisiondetection
{



// Implementation of generic continuous collision detection
//
// Original Author: Ziqiu ZENG
//
class GenericCCD : public BaseCollision
{
public:
    GenericCCD(real alarm_distance, real minimum_separation, bool VF, bool EE, bool FV):
            _alarm_distance(alarm_distance), _minimum_separation(minimum_separation), _VF(VF), _EE(EE), _FV(FV){}

    void addObjectPair(RealSim::geometry::SBaseGeometry * meshA, RealSim::geometry::SBaseGeometry * meshB, real muVF, real muEE, real muFV)
    {
        _objectPairs.emplace_back(meshA, meshB);
        _muVF.emplace_back(muVF);
        _muEE.emplace_back(muEE);
        _muFV.emplace_back(muFV);
    }

    // detect collision
    void detect(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real dt) override;

    void detect_impl_v0(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real /*dt*/) ; // add BVH for collide
    void detect_impl_v1(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real /*dt*/) ; // add BVH for collide
    void detect_impl_v2(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real /*dt*/) ; // add boardphrase for multi-objects
    void detect_impl_v0_parallel(std::vector<RealSim::contact::ContactPair> &output, const MatX3R &pos0, const MatX3R &pos1, real /*dt*/) ; // add boardphrase for multi-objects

    static void doVFDetection(std::vector<RealSim::contact::ContactPair> &output,
                              const MatX3R &pi_t0,
                              const MatX3R &pi_t1,
                              const MatX3R &pj_t0,
                              const MatX3R &pj_t1,
                              const VecXI* vset,
                              const MatX3I* fset,
                              real mu,
                              bool first_static,
                              bool second_static,
                              real alram_distance,
                              real minimum_separation);

    static void doEEDetection(std::vector<RealSim::contact::ContactPair> &output,
                              const MatX3R &pi_t0,
                              const MatX3R &pi_t1,
                              const MatX3R &pj_t0,
                              const MatX3R &pj_t1,
                              const MatX2I* eset0,
                              const MatX2I* eset1,
                              real mu,
                              bool first_static,
                              bool second_static,
                              real alram_distance,
                              real minimum_separation);

    static void doVFDetectionParallel(std::vector<RealSim::contact::ContactPair> &output,
                                      const MatX3R &pi_t0,
                                      const MatX3R &pi_t1,
                                      const MatX3R &pj_t0,
                                      const MatX3R &pj_t1,
                                      const VecXI* vset,
                                      const MatX3I* fset,
                                      real mu,
                                      bool first_static,
                                      bool second_static,
                                      real alram_distance,
                                      real minimum_separation);

    static void doEEDetectionParallel(std::vector<RealSim::contact::ContactPair> &output,
                                      const MatX3R &pi_t0,
                                      const MatX3R &pi_t1,
                                      const MatX3R &pj_t0,
                                      const MatX3R &pj_t1,
                                      const MatX2I* eset0,
                                      const MatX2I* eset1,
                                      real mu,
                                      bool first_static,
                                      bool second_static,
                                      real alram_distance,
                                      real minimum_separation);

    static void VFDetection(RealSim::contact::ContactPair &output,
                            const MatX3R &pi_t0,
                            const MatX3R &pi_t1,
                            const MatX3R &pj_t0,
                            const MatX3R &pj_t1,
                            Index v,
                            Index f0,
                            Index f1,
                            Index f2,
                            real mu,
                            bool first_static,
                            bool second_static,
                            real alram_distance,
                            real minimum_separation);

    static void EEDetection(RealSim::contact::ContactPair &output,
                            const MatX3R &pi_t0,
                            const MatX3R &pi_t1,
                            const MatX3R &pj_t0,
                            const MatX3R &pj_t1,
                            Index ea0,
                            Index ea1,
                            Index eb0,
                            Index eb1,
                            real mu,
                            bool first_static,
                            bool second_static,
                            real alram_distance,
                            real minimum_separation);

protected:
    std::vector<std::pair<RealSim::geometry::SBaseGeometry *, RealSim::geometry::SBaseGeometry *>> _objectPairs;
    std::vector<real> _muVF, _muEE, _muFV;
    bool _VF, _EE, _FV;

    real _minimum_separation;
    real _alarm_distance;
};

struct CCDVFDetectionInRange {
    std::vector<RealSim::contact::ContactPair> &output;
    const MatX3R &pi_t0;
    const MatX3R &pi_t1;
    const MatX3R &pj_t0;
    const MatX3R &pj_t1;
    const VecXI *vset;
    const MatX3I *fset;
    real mu;
    bool first_static;
    bool second_static;
    real alram_distance;
    real minimum_separation;
    bool order;

public:
    CCDVFDetectionInRange(std::vector<RealSim::contact::ContactPair> &_output,
                       const MatX3R &_pi_t0,
                       const MatX3R &_pi_t1,
                       const MatX3R &_pj_t0,
                       const MatX3R &_pj_t1,
                       const VecXI *_vset,
                       const MatX3I *_fset,
                       real _mu,
                       bool _first_static,
                       bool _second_static,
                       real _alram_distance,
                       real _minimum_separation,
                       bool _order)
            :output(_output),
             pi_t0(_pi_t0),
             pi_t1(_pi_t1),
             pj_t0(_pj_t0),
             pj_t1(_pj_t1),
             vset(_vset),
             fset(_fset),
             mu(_mu),
             first_static(_first_static),
             second_static(_second_static),
             alram_distance(_alram_distance),
             minimum_separation(_minimum_separation),
             order(_order){}

    void operator()(const tbb::blocked_range<size_t>& r) const;
};


struct CCDEEDetectionInRange {
    std::vector<RealSim::contact::ContactPair> &output;
    const MatX3R &pi_t0;
    const MatX3R &pi_t1;
    const MatX3R &pj_t0;
    const MatX3R &pj_t1;
    const MatX2I *eset0;
    const MatX2I *eset1;
    real mu;
    bool first_static;
    bool second_static;
    real alram_distance;
    real minimum_separation;
    bool order;

public:
    CCDEEDetectionInRange(std::vector<RealSim::contact::ContactPair> &_output,
                       const MatX3R &_pi_t0,
                       const MatX3R &_pi_t1,
                       const MatX3R &_pj_t0,
                       const MatX3R &_pj_t1,
                       const MatX2I *_eset0,
                       const MatX2I *_eset1,
                       real _mu,
                       bool _first_static,
                       bool _second_static,
                       real _alram_distance,
                       real _minimum_separation,
                       bool _order)
            :output(_output),
             pi_t0(_pi_t0),
             pi_t1(_pi_t1),
             pj_t0(_pj_t0),
             pj_t1(_pj_t1),
             eset0(_eset0),
             eset1(_eset1),
             mu(_mu),
             first_static(_first_static),
             second_static(_second_static),
             alram_distance(_alram_distance),
             minimum_separation(_minimum_separation),
             order(_order){}

    void operator()(const tbb::blocked_range<size_t>& r) const;
};

}
