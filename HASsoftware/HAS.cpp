/*
       Squishy sampler project, the RATTLE Accelerated Sampler approach.
       See main.cpp for more information.
       

*/

#include <iostream>
#include <blaze/Math.h>   // numerical linear algebra and matrix/vector classes
#include <random>         // for the random number generators
#include <cmath>          // defines exp(..), M_PI
#include <string>
#include "model.hpp"
#include "HAS.hpp"
using namespace std;
using namespace blaze;


void HASampler(      vector<double>& chain,        /* Position Samples output from the MCMC run, pre-allocated, length = d*T */
                     struct SamplerStats *stats,     /* statistics about different kinds of rections                          */
                     size_t T,                       /* number of MCMC steps        */
                     double eps,                     /* squish parameter            */
                     double dt,                      /* time step size in RATTLE integrator                                    */
                     double gamma,                   /* friction coefficient for thermostat part in Langevin dynamics          */
                     int Nsoft,                      /* number of Soft Moves: Gaussian Metropolis move to resample position q  */
                     int Nrattle,                    /* number of Rattle steps         */
                     DynamicVector<double>& q0,      /* starting position              */
                     DynamicVector<double>& p0,      /* starting momentum              */
                     Model M,                        /* evaluate q(x) and grad q(x)    */
                     double sq,                      /* isotropic gaussian standard dev. for Position Soft move         */
                     double sp,                      /* isotropic gaussian standard dev. for Momentum Soft move         */
                     double neps,                    /* convergence tolerance for Newton projection                     */
                     double rrc,                     /* closeness criterion for the reverse check                       */
                     int   itm,                      /* maximum number of Newton iterations per projection              */
                     bool gradRATTLE,                /* if True, use grad V in RATTLE steps; if False, set grad V = 0 in RATTLE steps */
                     bool LangevinROLL,              /* if True, use the Langevin ROLL algorithm; if False, use plain ROLL            */
               mt19937 RG) {                   /* random generator engine, already instantiated      */
   
   DynamicVector<double, columnVector> xiDummy = M.xi(q0);// qStart is used only to learn m
   int d  = q0.size();      // infer dimensions, d = dimension of ambient space
   int m  = xiDummy.size();  // m = number of constraints
   int n  = d - m;           // n = dimension of constraint Surface (ie, dimension of its tangent space)
   
   DynamicVector<double, columnVector> Z(d);       // standard gaussian in ambient space R^d
   DynamicVector<double, columnVector> q(d);       // current position sample
   DynamicVector<double, columnVector> p(d);       // current momentum sample
   DynamicVector<double, columnVector> qn(d);      // proposed new position sample, also used for intermediate step in RATTLE integrator
   DynamicVector<double, columnVector> pn(d);      // proposed / new momentum sample, also used for intermediate step in RATTLE integrator
   DynamicVector<double, columnVector> qr(d);      // position for reverse step
   DynamicVector<double, columnVector> pr(d);      // momentum for reverse step
   DynamicVector<double, columnVector> v(d);       // tangent step proposal for momentum resampling
   DynamicVector<double, columnVector> vr(d);      // reverse tangent step for momentum resampling
   
   
   DynamicVector<double, columnVector> z(m);          // level z = S_xi(q)
   DynamicVector<double, columnVector> zn(m);         // level zn = S_xi(qn)
   DynamicVector<double, columnVector> r(m);          // residual in Newton projection step
   DynamicMatrix<double, columnMajor> gtq2gq1(m,m);   // grad(xi)^t(q2)*grad(xi)(q1), used in Newton position projection
   DynamicMatrix<double, columnMajor> gtqrgq2(m,m);   // grad(xi)^t(qr)*grad(xi)(q2), used in Newton position projection
   DynamicMatrix<double, columnMajor> gtygy(m,m);     // grad(xi)^t(y)*grad(xi)(y), used in momentum projection
   
   DynamicVector<double, columnVector> a(m);       // coefficient in Newton's iteration
   DynamicVector<double, columnVector> da(m);      // increment of a in Newton's iteration
   DynamicVector<double, columnVector> gVqr(d);    // gradient of V at qr; used for RATTLE integrator
   DynamicMatrix<double, columnMajor> gxiqr(d,m);  // gradient matrix of constraint functions at qr; used for RATTLE integrator
   
   DynamicVector<double, columnVector> q1(d);      // intermediate position storage for RATTLE steps
   DynamicVector<double, columnVector> p1(d);      // intermediate momentum storage for RATTLE steps
   DynamicVector<double, columnVector> q2(d);      // intermediate position storage for RATTLE steps
   DynamicVector<double, columnVector> p2(d);      // intermediate momentum storage for RATTLE steps
   DynamicVector<double, columnVector> gVq1(d);    // gradient of V at q1
   DynamicVector<double, columnVector> gVq2(d);    // gradient of V at q2
   DynamicMatrix<double, columnMajor> gxiq1(d,m);  // gradient matrix of constraint functions at q1
   DynamicMatrix<double, columnMajor> gxiq2(d,m);  // gradient matrix of constraint functions at q2
   
   double Vq;                                      // potential evaluation: V(q)
   double Vqn;                                     // potential evaluation: V(qn)
   DynamicVector<double, columnVector> gVq(d);    // gradient of potential evaluation: grad V(q)
   DynamicVector<double, columnVector> gVqn(d);    // gradient of potential evaluation: grad V(qn)
   DynamicVector<double, columnVector> xiq(m);     // constraint function values at q
   DynamicVector<double, columnVector> xiqn(m);    // constraint function values at qn
   DynamicMatrix<double, columnMajor> gxiq(d,m);   // gradient matrix of constraint functions
   DynamicMatrix<double, columnMajor> gxiqn(d,m);  // gradient matrix of constraint functions
   DynamicMatrix<double, columnMajor> Tq(d,n);     // for basis of tangent space
   DynamicMatrix<double, columnMajor> Tqn(d,n);    // for basis of tangent space
   
   DynamicVector<double, columnVector> R(n);       // isotropic Gaussian of variance 1 sampled in q-tangent space of level surface S_xi(q)
   DynamicMatrix<double, columnMajor> Agxiq(d,d);  // augmented gxi(q) used for 'full' SVD decomposition
   DynamicMatrix<double, columnMajor> Agxiqn(d,d); // augmented gxi(qn) used for 'full' SVD decomposition where qn = position proposal
   DynamicMatrix<double, columnMajor>  U;          // U contains left singular vectors, size (d x d)
   DynamicVector<double, columnVector> s;          // vector contains m singular values, size m
   DynamicMatrix<double, columnMajor>  Vtr;        // V^t where V contains right singular vectors, size (m x m)
   
   double c1;   // coefficient 1 in thermostat momentum step
   double c2;   // coefficient 2 in thermostat momentum step
   double c3;   // coefficient 3 in thermostat momentum step
   
   double Uq;      // |xi(q)|^2
   double Uqn;     // |xi(qn)|^2
   double A;       //  Metropolis ratio
   double detq, detqn;    // detq = sqrt( det(gtg) ) = det(S) where S is the singular value matrix in reduced SVD of ( gxi * gxi^t )
   double v_sqn, vr_sqn;  // |v|^2, |vrev|^2  used in Metropolis-Hastings check for Soft move
   double p_sqn, pn_sqn;  // |p|^2, |pn|^2  used in both Metropolis-Hastings checks
   
   int       softFlag;                       // a flag describing where in the SOFT move step you are
   const int Met_rej_soft               = 1;         // the proposed qn was rejected by Metropolis
   const int Met_acc_soft               = 2;         // the proposed qn was accepted by Metropolis
   const int accept_qn_soft             = 3;         // accepted the new point
   
   int       rtFlag;                         // a flag describing where in the "RATTLE" move you are
   const int starting_proj_q2             = 1;            // possible values: have not started the q projection -- forward RATTLE step
   const int q2_proj_worked               = 2;            // the Newton iteration found qn on the surface xi(qn)=xi(q)
   const int RATTLE_sequence_worked       = 3;            // RATTLE q-projections were ALL successful
   const int RATTLE_sequence_worked_rev   = 4;            // RATTLE q-projections were ALL successful
   const int starting_proj_qr             = 5;            // possible values: have not started the q projection -- reverse RATTLE step
   const int qr_proj_worked               = 6;            // the Newton iteration found qn on the surface xi(qn)=xi(q)
   const int qr_proj_failed               = 7;            // the qn projection Newton iteration ended without success
   const int q_reverse_check_worked       = 8;     // the POSITION reverse check iteration found q
   const int q_reverse_check_failed       = 9;     // the POSITION reverse check iteration failed or found a point that isn't q
   const int p_reverse_check_failed       = 10;     // the MOMENTUM reverse check found a point that isn't p
   const int reverse_check_worked         = 11;    // the overall reverse check worked
   const int reverse_check_failed         = 12;    // either (1) POSITION reverse projection did not converge or (2) found point that is not (q,p)
   const int Met_rej_rt                   = 13;          // the proposed qn was rejected by Metropolis
   const int Met_acc_rt                   = 14;          // the proposed 1n was accepted by Metropolis
   const int accept_qn_rt                 = 15;          // accepted the new point
   
   
   normal_distribution<double>       SN(0.,1.);   // standard normal (mean zero, variance 1)
   uniform_real_distribution<double> SU(0.,1.);   // standard uniform [0,1]
   
   stats-> SoftSample                        = 0;
   stats-> SoftSampleAccepted                = 0;
   stats-> SoftRejectionMetropolis           = 0;
   
   stats-> HardSample                        = 0;
   stats-> HardSampleAccepted                = 0;
   stats-> HardFailedProjection_q2           = 0;     // Rattle move: failed forward q-projection
   stats-> HardFailedProjection_qr           = 0;     // Rattle move: failed reverse q-projection
   stats-> HardRejectionReverseCheck_q       = 0;
   stats-> HardRejectionReverseCheck_p       = 0;
   stats-> HardRejectionReverseCheck         = 0;
   stats-> HardRejectionMetropolis           = 0;
   
//    Setup for the MCMC iteration: get values at the starting point
   
   // Initialize V gradients to 0 (in case gradRATTLE == false):
   
   gVq  = 0.;
   gVqn = 0.;
   gVq1 = 0.;
   gVq2 = 0.;
   gVqr = 0.;
   
   // Update these at the end of each move if proposal is accepted:
   
   q   = q0;         // starting position
   p   = p0;         // must be on T_q0
   Vq  = M.V(q);     // potential evaluated at starting point
   if (gradRATTLE){  // if true ..
      gVq = M.gV(q);   // .. evaluate gradient of potential evaluated at starting point (used in RATTLE steps only)
   }
   xiq  = M.xi(q);   // constraint function evaluated at starting point
   gxiq = M.gxi(q);  // gradient of constraint function at starting point
   
   int Nsample = -1; // we will increment this varialble each time a new sample is stored in chain[]
   int nsteps = 0;   // will be used to count how many RATTLE forward time-steps are taken before a failed projection
   
   
   
   //    Start MCMC loop
   
   for (unsigned int iter = 0; iter < T; iter++){
      
      //cout << "------" << endl;
      //cout << q << endl;
      //cout << p << endl;
      //cout << xiq << endl;
      //cout << trans( gxiq ) * p << endl;
      
      //----------------------------------------------Position-Momentum resampling Metropolis-Hastings---------------------------------------------
      
      for (unsigned int i = 0; i < Nsoft; i++){

         stats-> SoftSample++;     // one soft move
         
         // Draw proposal qn: isotropic gaussian ( std = sq ) in ambient space
         for (unsigned int k = 0; k < d; k++){  // Sample Isotropic Standard Gaussian
            Z[k] = SN(RG);
         }
         qn   = q + sq*Z;      // Position proposal: Isotropic gaussian with mean zero and covariance sm^2*Id
         
         // Draw proposal pn = p + v + gxi(q)da in Tqn, where v isotropic gaussian ( std = sp ) in ambient space
         for ( unsigned int k = 0; k < n; k++){   // Isotropic Gaussian, not tangent
            R[k] = SN(RG);
         }  // used for tangent step v = Tq * R
         
         //       Compute Tq =  basis for tangent space at q. To do so, calculate Full SVD of gxiq = U * S * V^t. Then,
         //       .. Tq = last d-m columns of U in the Full SVD for gxiq
         
         Agxiq = M.Agxi(gxiq);        // add d-n column of zeros to gxiq to get full SVD, needed to get Tq = last d-n columns of U
         svd( Agxiq, U, s, Vtr);        // Computing the singular values and vectors of gxiq
         
         //    Build Tq = matrix for tangent space basis at q
         for ( unsigned long int i = m; i < d; i++){
            unsigned long int k = i-m;
            column(Tq,k) = column(U,i);
         }      // end of calculation for Tq
         
         v       = sp * Tq * R;      // tangent proposal in T_q
         pn      = p + v;            // step in current tangent space T_q
         gxiqn   = M.gxi(qn);        // compute gxi(qn);
         gtq2gq1 = trans( gxiqn ) * gxiq;    // compute matrix for tangent step projection gxi(qn)^t gxi(q)
         r       = - trans( gxiqn ) * pn;    // right hand side of linear system for projection
         solve( gtq2gq1, da, r);             // solve linear system for projection onto Tqn
         pn = pn + gxiq*da;         // Momentum proposal in Tqn
         
         // find reverse tangent step vr
         gtygy = trans( gxiqn ) * gxiqn;   // compute matrix for tangent step projection gxi(qn)^t gxi(qn)
         r     = trans( gxiqn ) * p;       // right hand side of linear system for reverse projection
         solve( gtygy, da, r);             // solve linear system for projection onto Tqn
         vr    = p - pn - gxiqn*da;        // reverse tangent step
         
         // Do the metropolis detail balance check
         
         Uq      = sqrNorm( xiq );          // |xi(q)|^2
         Vqn     = M.V(qn);                 // V(qn)
         if (gradRATTLE){
            gVqn = M.gV(qn);   // grad V(qn)
         }
         xiqn    = M.xi(qn);                // evaluate xi(qn) (also used when processing accepted proposal)
         Uqn     = sqrNorm( xiqn );         // |xi(qn)|^2
         v_sqn   = sqrNorm( v );            // |v|^2
         vr_sqn  = sqrNorm( vr );           // |vrev|^2
         p_sqn   = sqrNorm( p );            // |p|^2
         pn_sqn  = sqrNorm( pn );           // |pn|^2
         
         A = exp( (Vq - Vqn) + 0.5*( ((Uq - Uqn) / (eps*eps)) + ((v_sqn - vr_sqn) / (sp*sp)) + (p_sqn - pn_sqn) ) );  // Metropolis ratio
         
         if ( SU(RG) > A ){      // Accept with probability A,
            softFlag = Met_rej_soft;    // rejected
            stats-> SoftRejectionMetropolis++;
         }
         else{
            softFlag = Met_acc_soft;    // accepted
         }
         
         if ( softFlag == Met_acc_soft ) {     //  process an accepted proposal
            stats-> SoftSampleAccepted++;
            q   = qn;
            p   = pn;
            Vq  = Vqn;     // update potential evaluation
            gVq = gVqn;    // update gradient
            gxiq = gxiqn;  // update gradient
            xiq  = xiqn;   // update constraint function
         }
         else {                                       // process a rejected proposal
         }
         
         Nsample++;
         for ( int k = 0; k < d; k++){    // add sample to Schain here
            chain[ k + d * Nsample ]  = q[k];    // value of q[k] where q is iter-th position sample
         }
         
      } // end of Soft Metropolis move
      
      //------------------------------------------------------------Thermostat half step-------------------------------------------------------------
      
      if (LangevinROLL){
         
         for (unsigned int k = 0; k < d; k++){  // Sample Isotropic Standard Gaussian
            Z[k] = SN(RG);
         }
         
         gtygy = trans( gxiq ) * gxiq;          // compute matrix for momentum step projection: gxi(q)^t gxi(q)
         c1    = 1. - ( gamma * dt * 0.25 );
         c2    = sqrt( gamma * dt );
         r     = - trans( gxiq ) * ( c1 * p + c2 * Z );   // right hand side of linear system for projection
         solve( gtygy, da, r);                            // solve linear system for projection onto Tq
         
         c3 = 1. / ( 1. + ( gamma * dt * 0.25 ) );
         pn = c3 * ( c1 * p + c2 * Z + gxiq * da);        // thermostat momentum step
         
         p = pn;    // update momentum
      }
      
      //---------------------------------------------------------------RATTLE steps------------------------------------------------------------------
      
      if ( Nrattle  > 0 ){
         
         stats-> HardSample++;     // one Rattle move
         
         z     = xiq;        // need the level z = S_xi(q1), actually have xiq from isotropic Metropolis above, can re-use that !!
         q1     = q;         // initial position for RATTLE iteration
         gVq1   = gVq;
         gxiq1  = gxiq;
         p1     = p;         // initial momentum for RATTLE iteration:
         
         nsteps = 0;            // reset nsteps
         
         // Take "Nrattle" time steps using RATTLE integrator : (q, p) --> (q2, p2)
         
         for (unsigned int i = 0; i < Nrattle; i++){
            
            rtFlag = starting_proj_q2;  // start q2 projection
            
            //    First, project q1 + dt*p1 onto the constraint surface S_z
            //    Newton loop to find q2 = q + dt * p1 + dt * grad(xi)(q1)*a, with xi(q2)=xi(q)=z
            a = 0;       // starting coefficient
            p1    = p1 - 0.5 * dt * gVq1;
            q2    = q1 + dt * p1;    // initial guess = move in the tangent direction
            gxiq2 = M.gxi(q2);       // because these are calculated at the end of this loop
            for ( int ni = 0; ni < itm; ni++){
               r     = z - M.xi(q2);              // equation residual
               gtq2gq1 = trans( gxiq2 )*gxiq1;    // Newton Jacobian matrix
               solve( dt * gtq2gq1, da, r);       // solve the linear system; Note the * dt on LHS
               q2  +=  dt * gxiq1*da;             // take the Newton step;    Note the * dt when updating
               a  += da;                          // need the coefficient to update momentum later
               gxiq2 = M.gxi(q2);                 // constraint gradient at the new point (for later)
               if ( norm(r) <= neps ) {
                  rtFlag = q2_proj_worked;     // record that you found q2
                  nsteps++;          // increment number of successful projection steps in forward RATTLE, needed for later
                  break;                      // stop the Newton iteration
               }  // end of norm(r) check
            }   // end of Newton solver loop
            
            if ( rtFlag != q2_proj_worked ){  // as soon as there is one failed forward q-projection ...
               stats->HardFailedProjection_q2++;
               break;  //  ... we break the forward RATTLE loop and reject the proposal and
            }
            
            //  ... otherwise continue and project momentum p2 = p1 + grad(xi)(q1)*a - 0.5 * dt * gVq2    onto T_q2
            p1    += gxiq1*a;                  // update p1 with 2nd lagrange multiplier computed from position projection
            if (gradRATTLE){
               gVq2  = M.gV(q2);               // compute grad V(q2)
            }
            p2    = p1 - 0.5 * dt * gVq2;      // add    - 0.5 * dt * gVq2    term to get p2 to be projected
            gtygy = trans( gxiq2 )*gxiq2;
            r     = - trans( gxiq2 )*p2;
            solve( gtygy, a, r);         // carry out projection
            
            // Set the new state to be the one produced by RATTLE iteration above
            p2    = p2 + gxiq2*a;   // update p2 with 2nd lagrange multiplier computed from p2 projection
            // q2 already set at the end of Newton iteration
            // end of RATTLE integrator single step
            
            // re-initialize for next iteration
            q1 = q2;
            p1 = p2;
            gVq1 = gVq2;
            gxiq1 = gxiq2;
            
            if ( nsteps == Nrattle ){  // If we had Nrattle successfull forward RATTLE projection ...
               rtFlag = RATTLE_sequence_worked;
               qn = q2;   // ... save the proposal for Metropolis check below
               pn = p2;
               gVqn = gVq2;
               gxiqn = gxiq2;
            }
         } // end of forward RATTLE loop
         
         if ( rtFlag == RATTLE_sequence_worked ){  // if forward RATTLE steps were all successful --> do the REVERSE CHECK !!!! :
            
            nsteps = 0;            // reset nsteps
            
            // (1) apply RATTLE integrator "Nrattle" times to starting from (q2, -p2) to get (qr, pr)
            
            p2    = - p2;   // apply first ** Momentum Reversal ** for reverse move / check
            
            for (unsigned int i = 0; i < Nrattle; i++){ // Take Nrattle reverse time steps using RATTLE integrator :
               
               rtFlag = starting_proj_qr;    // start qr projection
               //    Project q2 + dt*p2 onto the constraint surface S_z, check if the result is = q
               //    Newton loop to find qr = q2 + dt*p2 + dt*grad(xi)(q2)*a with xi(qr) = xi(q2)= xi(q1) = z
               a = 0;                     // starting coefficient
               p2    = p2 - 0.5 * dt * gVq2;
               qr    = q2 + dt * p2;       // initial guess = move in the tangent direction
               gxiqr = M.gxi(qr);           // because these are calculated at the end of this loop
               for ( int ni = 0; ni < itm; ni++){
                  r     = z - M.xi(qr);            // equation residual (we project onto same level set as for starting point q)
                  gtqrgq2 = trans( gxiqr )*gxiq2;  // Newton Jacobian matrix;
                  solve( dt * gtqrgq2, da, r);     // solve the linear system; Note the * dt on LHS  ...!!
                  qr +=  dt * gxiq2*da;            // take the Newton step;    Note the * dt when updating ...!!
                  a  += da;                        // need the coefficient to update momentum later
                  gxiqr = M.gxi(qr);               // constraint gradient at the new point (for later)
                  if ( norm(r) <= neps ) {    // If Newton step converged ...
                     rtFlag = qr_proj_worked;
                     nsteps++;
                     break;                   // stop the Newton iteration, it converged
                  }   // end of norm(r) check
               }    // end of Newton solver loop
               
               if ( rtFlag != qr_proj_worked ){   // as soon as there is one failed reverse q-projection ...
                  stats->HardFailedProjection_qr++;
                  break;
               }
               
               //  ... otherwise continue and project momentum pr = Proj ( p2 + grad(xi)(q2)*a )  onto T_qr
               p2    += gxiq2*a;
               if (gradRATTLE){
                  gVqr  = M.gV(qr);
               }
               pr    = p2 - 0.5 * dt * gVqr;
               gtygy = trans( gxiqr )*gxiqr;
               r     = - trans( gxiqr )*pr;
               solve( gtygy, a, r);
               
               pr    = pr + gxiqr*a;
               // qr already set at the end of Newton iteration
               // end of RATTLE integrator single step
               
               // re-initialize for next iteration
               q2 = qr;
               p2 = pr;
               gVq2  = gVqr;
               gxiq2 = gxiqr;
               
               if ( nsteps == Nrattle ){  // If we had Nrattle successfull reverse RATTLE projection --> go on with reverse check
                  rtFlag = RATTLE_sequence_worked_rev;
               }
            } // end of reverse RATTLE loop
            
            if ( rtFlag == RATTLE_sequence_worked_rev ){  // if reverse RATTLE projections were ALL successfull, go on with reverse check
               
               // (2) Check whether (qr, -pr) = (q, p) ?
               
               pr = - pr;  // apply second  ** Momentum Reversal ** for reverse move / check
               
               if ( norm( qr - q ) < rrc ) {      // ... did reverse RATTLE go back to the right point?
                  rtFlag = q_reverse_check_worked;
               } else {
                  stats->HardRejectionReverseCheck_q++; // converged to the wrong point --> a failure
                  stats->HardRejectionReverseCheck++;   // so reject proposal
               }
               
               if ( rtFlag == q_reverse_check_worked) {
                  
                  if (norm( pr - p ) < rrc) {
                     rtFlag = reverse_check_worked;
                  } else{
                     rtFlag = p_reverse_check_failed;
                     stats->HardRejectionReverseCheck_p++;
                  }
               }
               
            } else {  // if one of the reverse RATTLE projections failed --> another kind of failure
               stats->HardRejectionReverseCheck_q++;
               stats->HardRejectionReverseCheck++;   // so reject proposal
            }
            
         }  // end of overall reverse check
         
         // Metropolis detailed balance check for (q,p) --> (qn, pn) :
         
         if ( rtFlag == reverse_check_worked ){  // do Metropolis check only if reverse check was successful
            
            Agxiq = M.Agxi(gxiq);        // add d-n column of zeros to gxiq to get full SVD, needed to get Tq = last d-n columns of U
            svd( Agxiq, U, s, Vtr);        // Computing the singular values and vectors of gxiq
            // multiply singular values to get detq = sqrt( det(gxiq^t gxiq) ) = det(S) (for RATTLE move later)
            detq = 1.;
            for ( unsigned long int i = 0; i < m; i++){
               detq *= s[i];    // detq = sqrt( det(gxiq^t gxiq) ) = det(S)
            }
            
            Agxiqn = M.Agxi(gxiqn);          // add d-n column of zeros to gxiq to get full SVD, needed to get Tqn = last d-n columns of U
            svd( Agxiqn, U, s, Vtr);        // Computing the singular values and vectors of gxiqn
            // multiply singular values to get detqn = r(qn) = sqrt( det(gxiy^t gxiy) ) = det(S)
            detqn = 1.;
            for ( unsigned long int i = 0; i < m; i++){
               detqn *= s[i];    // detqn = sqrt( det(gxiy^t gxiy) ) = det(S)
            }
            
            Vqn    = M.V(qn);        // V(qn)   (not yet evaluated, since it was not necessary in RATTLE steps)
            p_sqn  = sqrNorm( p );   // |p|^2  for M-H ratio
            pn_sqn = sqrNorm( pn );  // |pn|^2 for M-H ratio
            
            // NOTE: Here can add V(q) and V(qn) to M-H ratio, for now assume V=0
            
            A  = exp( Vq - Vqn + .5*( p_sqn - pn_sqn ) ); //  part of the Metropolis ratio
            A *= ( detq / detqn );    // since r(q)/r(qn) = detq/detqn
            
            if ( SU(RG) > A ){      // Accept with probability A,
               stats->HardRejectionMetropolis++;
            }
            else{
               rtFlag = Met_acc_rt;                       // accepted
            }    // Metropolis rejection step done
         } // end of overall Metropolis check
         
         if ( rtFlag ==  Met_acc_rt) {     //  process an accepted proposal
            q   = qn;
            p   = pn;
            Vq  = Vqn;        // update potential (evaluated in the Metropolis check)
            gVq = gVqn;       // update gradient (was already evaluated in RATTLE steps)
            gxiq = gxiqn;     // update gradient (xiq stays the same)
            stats-> HardSampleAccepted++;
         }
         else {         // process a rejected proposal
            
            p = - p;   // very important: apply ** MOMENTUM REVERSAL ** in the rejection step !!
            
         }
      }  // end of RATTLE move
      
      //------------------------------------------------------------Thermostat half step-------------------------------------------------------------
      
      if (LangevinROLL){
         
         for (unsigned int k = 0; k < d; k++){  // Sample Isotropic Standard Gaussian
            Z[k] = SN(RG);
         }
         
         gtygy = trans( gxiq ) * gxiq;          // compute matrix for momentum step projection: gxi(q)^t gxi(q)
         r     = - trans( gxiq ) * ( c1 * p + c2 * Z );   // right hand side of linear system for projection
         solve( gtygy, da, r);                            // solve linear system for projection onto Tq
         
         pn = c3 * ( c1 * p + c2 * Z + gxiq * da);        // thermostat momentum step
         
         p = pn;    // update momentum
      }
      
         
      if ( Nsoft == 0 ){  // ONLY FOR DEBUGGING : store the sample when number of Soft moves = 0
         Nsample++;
         for ( int k = 0; k < d; k++){
            chain[ k + d * Nsample] = q[k];
         }
      } // end of DEBUGGING secton


   } // end of MCMC loop
   
} // end of sampler





















