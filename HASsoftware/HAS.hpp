/*
       Squishy sampler project, the RATTLE Accelerated Sampler approach.
       See main.cpp for more information.
       

*/
//
//  HAS.hpp
//

#ifndef HAS_hpp
#define HAS_hpp

#include <iostream>
#include <blaze/Math.h>
#include <random>
#include <string>
#include "model.hpp"
using namespace std;
using namespace blaze;



struct SamplerStats{                     /* Acceptance/rejection data from the sampler */
   int SoftSample;
   int SoftSampleAccepted;
   int SoftRejectionMetropolis;
   int HardSample;
   int HardSampleAccepted;
   int HardFailedProjection_q2;          // Hard move: failed projection in RATTLE time step
   int HardFailedProjection_qr;          // Hard move: failed projection in reverse RATTLE time step
   int HardRejectionReverseCheck_q;
   int HardRejectionReverseCheck_p;
   int HardRejectionReverseCheck;
   int HardRejectionMetropolis;
};


void HASampler(   vector<double>& chain,         /* Samples output from the MCMC run, pre-allocated, length = d*T */
                  struct SamplerStats *stats,     /* statistics about different kinds of rections                  */
                  size_t T,                       /* number of MCMC steps        */
                  double eps,                     /* squish parameter            */
                  double dt,                      /* time step size in RATTLE integrator                                    */
                  double Nsoft,                   /* number of Soft Moves: Gaussian Metropolis move to resample position q  */
                  double Nrattle,                 /* number of Rattle steps      */
                  DynamicVector<double>& q0,      /* starting position           */
                  DynamicVector<double>& p0,      /* starting momentum           */
                  Model M,                        /* evaluate q(x) and grad q(x) */
                  double sq,                      /* isotropic gaussian standard dev. for position Soft move     */
                  double sp,                      /* isotropic gaussian standard dev. for momentum Soft move     */
                  double neps,                    /* convergence tolerance for Newton projection        */
                  double rrc,                     /* closeness criterion for the reverse check          */
                  int   itm,                      /* maximum number of Newton iterations per projection */
                  mt19937 RG);                    /* random generator engine, already instantiated      */
 







#endif /* HAS.hpp */

