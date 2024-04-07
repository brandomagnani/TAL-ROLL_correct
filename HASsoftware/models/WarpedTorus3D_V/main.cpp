/*
       Squishy sampler project, the RATTLE Accelerated Sampler approach.
       See main.cpp for more information.
       
       The correctness checks below are adapted from Jonathan Goodman's
       implementation of the Foliation sampler
*/
//
//  3D Warped Torus model
//
//  main.cpp
//
#include <cstdio>
#include <iostream>
#include <fstream>                // write the output to a file
#include <blaze/Math.h>
#include <random>                 // contains the random number generator
#include <cmath>                  // defines exp(..), M_PI
#include <numeric>
#include <chrono>
#include "HAS.hpp"
#include "model.hpp"
using namespace std;
using namespace blaze;



int main(int argc, char** argv){
   
   const int d = 3;       // dimension of ambient space
   const int m = 2;       // number of constraint functions

   
   //  Set up the specific model problem -- a surface defined by intersection of m spheres
   
   DynamicVector<double, columnVector> ck(d);         // Center of sphere k
   DynamicMatrix<double, columnMajor>  c(d,m);        // Column k is the center of sphere k
   DynamicVector<double, columnVector> s(d);          // dimensional radius numbers for the ellipsoid

   double r;                                          // r = radius of sphere
   
   ck[0] = 0.;                   // first center: c_0 = ( 0, 0, 1)
   ck[1] = 0.;
   ck[2] = 1.;
   column( c, 0UL) = ck;         // 0UL means 0, as an unsigned long int
   
   ck[0] = 0.;                   // second center: c_0 = ( 0,-1, 0, )
   ck[1] = -1.;
   ck[2] =  0.;
   column( c, 1UL) = ck;
   
   r = sqrt(2.);                // The distance between centers is sqrt(2)
   s[0] = sqrt(2.);
   s[1] = sqrt(3.);
   s[2] = sqrt(5.);
   
   // Create a mass DynamicVector of size d filled with ones
      
   DynamicVector<double, columnVector> masses(d, 1.0);
   double ms = 1.0;
   masses[2] *= ms;
   
   // copy the parameters into the instance model
   
   Model M( d, m, r, s, c, masses);   // instance of model " 3D Warped Torus "



   cout << "--------------------------------------" << endl;
   cout << "\n  Running model: " << M.ModelName() << "\n" << endl;
   
   
//        Inputs for the SoftSampler
   
   
   DynamicVector<double, columnVector> q( d); // starting Position for sampler
   DynamicVector<double, columnVector> p( d); // starting Momentum for sampler, on T_q

   
   
// Starting point on Cotangent Bundle T S_0 :
   
   q[0] =  1.15826364700193;                 // ON 3D Warped torus level S_0
   q[1] = -0.0170130472486141;
   q[2] =  0.18874425718081;

   p[0] = -0.061452943055924;
   p[1] = 0.119011589305959;
   p[2] = -0.0902347507341671;

   auto q_tilde = M.M_sqrt * q;
   auto p_tilde = M.M_sqrt_inv * p;

   size_t T      = 2e6;          // number of MCMC steps
   double neps   = 1.e-10;       // convergence tolerance for Newton projection
   double rrc    = 1.e-8;        // closeness criterion for the reverse check
   int itm       = 6;            // maximum number of Newtons iterations
   
   int debug_off = 0;                // if = 1, then prints data on Metropolis ratio for Off move
   int debug_on  = 0;                // if = 1, then prints data on Metropolis ratio for On  move
   
   int integrate = 1;                // if = 1, then it does the integration to find marginal density for x[0]
   

// --------------------------------------------KEY PARAMETERS FOR SAMPLER---------------------------------------------------------
   
   
   double beta   = 1000.0;                      // squish parameter, beta = 1 / 2*eps^2
   double eps    = 1.0 / sqrt(2.0*beta);        // squish parameter
   
   double gamma_q = 1.;        // friction coefficient for thermostat part in Langevin dynamics
   double T_q     = 1.;        // physical temperature
   double beta_q  = 1. / T_q;  // physical variables inverse temperature
   
   double gamma_s = 1.;       // artificial friction coefficient for (extended var) thermostat part in Langevin dynamics
   double T_s     = T_q.;     // artificial temperature for extended variables s, must be large to overcome energy barriers
   double beta_s  = 1. / T_s; // artificial inverse temperature
   
   int Nsoft = 1;          // number of Soft moves for MCMC step
   int Nrattle = 3;        // number of RATTLE integrator time steps for each MCMC step
   
   double kq  = 0.7;       // factor for Soft Position proposal standard dev.
   double sq  = kq*eps;    // standard dev. for Soft Position proposal
   
   double kp  = 1.0;       // factor for Soft Momentum proposal standard dev.
   double sp  = kp*eps;    // standard dev. for Soft Momentum proposal
   
   double dt  = 0.5;       // time step size in RATTLE integrator
   
   bool gradRATTLE   = true;  // if True, use grad V in RALLTE steps; if False, set grad V = 0 in RATTLE steps
   bool LangevinROLL = true;  // if True, use the Langevin ROLL algorithm; if False, use plain ROLL
   
   
// -------------------------------------------------------------------------------------------------------------------------------
   
   
   size_t size_factor = Nsoft + Nrattle;  // multiplies T below
   size_t T_chain = size_factor * T;      // length of chain, needs to contain all samples
   vector<double> chain( d * T_chain);       // chain[k+d*l] is the value of q[k] where q is l-th sample
   unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
   mt19937 RG(seed);        // Mersenne twister random number generator
   SamplerStats stats;
   
   
   auto start = chrono::steady_clock::now();
   HASampler(chain, &stats, T, eps, dt, gamma_q, gamma_s, beta_q, beta_s, Nsoft, Nrattle, q, p, M, sq, sp, neps, rrc, itm, gradRATTLE, LangevinROLL, RG);
   auto end = chrono::steady_clock::now();
   
   int Ts;
   if ( Nsoft > 0 ){
      Ts = stats.SoftSample;    // number of Soft samples
   } else {
      Ts = stats.HardSample;
   }
   
   int Tr = stats.HardSample;    // number of Rattle samples
   double As = 0.;                  // Pr of Acceptance of Soft sample
   double Ar = 0.;                  // Pr of Acceptance of Rattle sample
   
   
   if (stats.SoftSample > 0) {  // Set Pr of Acceptance of Soft sample
      As    = (double) stats.SoftSampleAccepted / (double) stats.SoftSample;
   }
   if (stats.HardSample > 0) {  // Set Pr of Acceptance of Rattle sample
      Ar    = (double) stats.HardSampleAccepted / (double) stats.HardSample;
   }


   cout << " beta = " << beta << endl;
   cout << " eps = " << eps << endl;
   cout << " Elapsed time : " << chrono::duration_cast<chrono::seconds>(end - start).count() << " sec" << endl;
   cout << " " << endl;
   cout << " Soft   move Metropolis Rejection            : " << stats.SoftRejectionMetropolis << endl;
   cout << " " << endl;
   cout << " RATTLE move POSITION projection failures    : " << stats.HardFailedProjection_q2 << endl;
   cout << " Rattle move POSITION reverse check failures : " << stats.HardRejectionReverseCheck_q << endl;
   cout << " Rattle move MOMENTUM reverse check failures : " << stats.HardRejectionReverseCheck_p << endl;
   cout << " Rattle move reverse check failures          : " << stats.HardRejectionReverseCheck << endl;
   cout << " "              << endl;
   cout << " RATTLE Metropolis rejection                 : " << stats.HardRejectionMetropolis << endl;
   cout << " "              << endl;
   cout << " Number of Soft   samples = " << Ts    << endl;
   cout << " Number of Rattle samples = " << Tr    << endl;
   cout << " " << endl;
   cout << " Soft   sample Acceptance Pr = " << As   << endl;
   cout << " RATTLE sample Acceptance Pr = " << Ar   << endl;
   cout << " " << endl;

   cout << "--------------------------------------" << endl;

   
//  setup for data analysis:
   
   int bin;
   char OutputString[200];
   int  StringLength;
   
//   NO ANGLE CHECK
   
   double Lx = -3.;   // left x boundary for histogram check
   double Rx =  3.;   // right x boundary for histogram check
   int nx   = 100;       // number of x1 values for the PDF and number of x1 bins
   
   double Ly =  Lx;  // left y boundary for integration / histogram check
   double Ry =  Rx;   // right y boundary for integration / histogram check
   int niy   = 500;     // number of integration points in y-direction
   
   double Lz = Lx;    // left z boundary for integration / histogram check
   double Rz = Rx;    // right z boundary for integration / histogram check
   int niz   = niy;      // number of integration points in z-direction
   
   
//   Histogram of the x coordinates q[0]=x, q[1]=y, q[2]=z
   
   vector<double> Ratio(nx);   // contains the random variables Ri_N

   if ( integrate == 1 ) {
      double x1    = .5;
      vector<double> fl(nx);  // approximate (un-normalized) true pdf for x[0]=x, compute by integrating out x,z variables
      vector<int>    Nxb(nx); // vector counting number of samples in each bin
      for ( bin = 0; bin < nx; bin++) Nxb[bin] = 0;
      double dx = ( Rx - Lx )/( (double) nx);
      for ( int i=0; i<nx; i++){
         x1 = Lx + dx*i + .5*dx;
         fl[i] = M.yzIntegrate( x1, Ly, Ry, Lz, Rz, eps, niy, niz);
      }
      
      int outliers = 0;       //  number of samples outside the histogram range
      for ( unsigned int iter = 0; iter < Ts; iter++){
         x1 = chain[ d*iter ];  // same as before, but with k=0 as we need q[0] of iter-th good sample
         bin = (int) ( ( x1 - Lx )/ dx);
         if ( ( bin >= 0 ) && ( bin < nx ) ){
            Nxb[bin]++;
         }
         else{
            outliers++;
         }
      }        // end of analysis loop
     
      double Z;
      cout << " " << endl;
      cout << "   bin    center      count      pdf          1/Z" << endl;
      for ( bin = 0; bin < nx; bin++){
         if ( Nxb[bin] > 0 ) {
            x1 = Lx + dx*bin + .5*dx;
            Z = ( (double) Nxb[bin]) / ( (double) Ts*dx*fl[bin]);
            Ratio[bin] = Z;
            StringLength = snprintf( OutputString, sizeof(OutputString)," %4d   %8.3f   %8d   %9.3e    %9.3e", bin, x1, Nxb[bin], fl[bin], Z);
            cout << OutputString << endl;
         }
      }
      
      // Compute relative standard error for 1/Z:
      
      // Define the range for the bins to be included in the computation
      int startBin = 31;
      int endBin = 68;

      // 1. Calculate the mean of Ratio for the specified bins
      double sum = std::accumulate(Ratio.begin() + startBin, Ratio.begin() + endBin + 1, 0.0);
      double mean = sum / (endBin - startBin + 1);

      // 2. Calculate the standard deviation of Ratio for the specified bins
      double sq_sum = std::inner_product(Ratio.begin() + startBin, Ratio.begin() + endBin + 1,
                                         Ratio.begin() + startBin, 0.0);
      double variance = sq_sum / (endBin - startBin + 1) - mean * mean;
      double std_dev = std::sqrt(variance);

      // 3. Calculate the Standard Error (SE)
      double SE = std_dev / std::sqrt(endBin - startBin + 1);

      // 4. Calculate the Relative Standard Error (RSE) and express it as a percentage
      double RSE = (SE / mean) * 100;
      
      cout << " " << endl;
      cout << " Mean 1/Z (Bins 31 to 68): " << mean << endl;
      cout << " Standard Error 1/Z (Bins 31 to 68): " << SE << endl;
      cout << " Relative Standard Error 1/Z (Bins 31 to 68): " << RSE << "%" << endl;
      cout << " " << endl;
      cout << " Number of outliers : " << outliers << endl;
      cout << " " << endl;
   }

   ofstream OutputFile ( "ChainOutput.py");
   OutputFile << "# data output file from an MCMC code\n" << endl;
   OutputFile << "import numpy as np" << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"eps = %10.5e", eps);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"kq = %10.5e", kq);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"kp = %10.5e", kp);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"sq = %10.5e", sq);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"sp = %10.5e", sp);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"gamma_q = %10.5e", gamma_q);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"gamma_s = %10.5e", gamma_s);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"beta_q = %10.5e", beta_q);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"beta_s = %10.5e", beta_s);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"dt = %10.5e", dt);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Nsoft = %10d", Nsoft);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Nrattle = %10d", Nrattle);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"T = %10d", (int)T);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Ts = %10d", (int)Ts);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Tr = %10d", (int)Ts);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"As = %6.3f", As);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"Ar = %6.3f", Ar);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"d = %10d", d);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"D0 = %6.3f", D0);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"a = %6.3f", a);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"kappa = %6.3f", kappa);
   OutputFile << OutputString << endl;
   StringLength = snprintf( OutputString, sizeof(OutputString),"lambda_ = %6.3f", lambda_);
   OutputFile << OutputString << endl;
   OutputFile << "ModelName = \"" << M.ModelName() << "\"" << endl;
   OutputFile.close();
   


// Write chain in binary format (in a file called "chain.bin"), much faster for Python to read when chain is very long
   size_t sample_size = Ts*d;
   ofstream ostream1("chain.bin", ios::out | ios::binary);
   ostream1.write((char*)&chain[0], sample_size * sizeof(double));
   ostream1.close();


   
}  // end of main

