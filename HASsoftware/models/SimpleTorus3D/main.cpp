/*
       Squishy sampler project, the RATTLE Accelerated Sampler approach.
       See main.cpp for more information.
       
       The correctness checks below are adapted from Jonathan Goodman's
       implementation of the Foliation sampler
*/
//
//
//  3D Simple Torus model
//
//
//  main.cpp
//
#include <cstdio>
#include <iostream>
#include <fstream>                // write the output to a file
#include <blaze/Math.h>
#include <random>                 // contains the random number generator
#include <cmath>                  // defines exp(..), M_PI
#include <chrono>
#include "HAS.hpp"
#include "model.hpp"
using namespace std;
using namespace blaze;


// ---------------------------------------- MAIN ----------------------------------------

int main(int argc, char** argv){
   
   const int d = 3;       // dimension of ambient space
   const int m = 2;       // number of constraint functions

   
   //  Set up the specific model problem -- a surface defined by intersection of m spheres
   
   DynamicVector<double, columnVector> ck(d);         // Center of sphere k
   DynamicMatrix<double, columnMajor>  c(d,m);        // Column k is the center of sphere k
   DynamicVector<double, columnVector> r(m);          // r_k = radius of sphere k


   
   ck[0] = 0.;                   // first center: c_0 = ( 0, 0, 1)
   ck[1] = 0.;
   ck[2] = 1.;
   column( c, 0UL) = ck;         // 0UL means 0, as an unsigned long int
   
   ck[0] = 0.;                  // second center: c_0 = ( 0,-1, 0, )
   ck[1] = -1.;
   ck[2] =  0.;
   column( c, 1UL) = ck;
   
   r[0] = sqrt(2.);                // The distance between centers is sqrt(2)
   r[1] = sqrt(2.);
   
   // copy the parameters into the instance model
   
   
   Model M(d, m, r, c);            // instance of model " 3D Intersecting Sphere "

   
   cout << "--------------------------------------" << endl;
   cout << "\n  Running model: " << M.ModelName() << "\n" << endl;
   
   
//        Inputs for the Sampler
   
   
   DynamicVector<double, columnVector> q( d); // starting Position for sampler
   DynamicVector<double, columnVector> p( d); // starting Momentum for sampler, on T_q
   
   
// Starting point on Cotangent Bundle T S_0 :

   q[0] = 1.20105824111462;                   // give value to starting point, on 3D Simple Torus model constraint S_0 ..
   q[1] = -0.669497937230318;                 // // .. r1 = sqrt(2), r2 = sqrt(2)
   q[2] = 0.669497937230317;

   p[0] = 0.0674610691512285;
   p[1] = 0.239013743714144;
   p[2] = -0.239013743714144;
   

   size_t T      = 2e7;     // number of MCMC steps
   double neps   = 1.e-10;       // convergence tolerance for Newton projection
   double rrc    = 1.e-8;        // closeness criterion for the reverse check
   int itm       = 6;            // maximum number of Newtons iterations
   
   int debug_off = 0;                // if = 1, then prints data on Metropolis ratio for Off move
   int debug_on  = 0;                // if = 1, then prints data on Metropolis ratio for On  move
   
   int integrate = 1;                // if = 1, then it does the integration to find marginal density for x[0]
   
   
// --------------------------------------------KEY PARAMETERS FOR SAMPLER---------------------------------------------------------
   
   
   double beta   = 1000.0;                      // squish parameter, beta = 1 / 2*eps^2
   double eps    = 1.0 / sqrt(2.0*beta);        // squish parameter
   
   int Nsoft = 1;          // number of Soft moves for MCMC step
   int Nrattle = 3;        // number of RATTLE integrator time steps for each MCMC step
   
   double kq  = 0.5;       // factor for Soft Position proposal std
   double sq  = kq*eps;    // std for Soft Position proposal
   double kp  = 1.0;       // factor for Soft Momentum proposal std
   double sp  = kp*eps;       // std for Soft Momentum proposal
   
   double kg    = 1.0;     // factor for gamma below
   double gamma = 1.0;     // friction coefficient for thermostat part in Langevin dynamics
   
   double dt  = 0.6;       // time step size in RATTLE integrator
   
   bool gradRATTLE   = true;  // if True, use grad V in RALLTE steps; if False, set grad V = 0 in RATTLE steps
   bool LangevinROLL = false;  // if True, use the Langevin ROLL algorithm; if False, use plain ROLL
   
// -------------------------------------------------------------------------------------------------------------------------------
   
   size_t size_factor = Nsoft + Nrattle;  // multiplies T below
   size_t T_chain = size_factor * T;      // length of chain, needs to contain all samples
   vector<double> chain( d * T_chain);       // chain[k+d*l] is the value of q[k] where q is l-th sample
   unsigned seed = 17U;     // seed for the random number generator -- 17 is the most random number?
   mt19937 RG(seed);        // Mersenne twister random number generator
   SamplerStats stats;
   
   
   auto start = chrono::steady_clock::now();
   HASampler(chain, &stats, T, eps, dt, gamma, Nsoft, Nrattle, q, p, M, sq, sp, neps, rrc, itm, gradRATTLE, LangevinROLL, RG);
   auto end = chrono::steady_clock::now();
   
   
   double Ts = stats.SoftSample;    // number of Soft samples
   double Tr = stats.HardSample;    // number of Rattle samples
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
   
   char OutputString[200];
   int  StringLength;
   

// Check angle of sample with respect to simmetry axis
   
   DynamicVector<double, columnVector> cc(3);          //  center of the circle is (0,-.5,.5)
   cc[0] = 0.;
   cc[1] =-.5;
   cc[2] = .5;
   
   DynamicVector<double, columnVector> n1(3), n2(3);   // two vectors normal to
   n1[0] =    1.;                                      // the axis between the two centers
   n1[1] =    0.;                                      // These vectors define the plane
   n1[2] =    0.;                                      // that the intersecton of the spheres is in
   n2[0] =    0.;
   n2[1] = - sqrt(.5);
   n2[2] =   sqrt(.5);
   DynamicVector<double, columnVector> t(3);          // vector from one center to the other
   t[0] = 0.;
   t[1] = .5;
   t[2] = .5;
   double ut, u1,  u2, theta;
   vector<double> thetas(Ts);
   
   int ntbins = 30;          // to bin the theta values
   int bin;
   vector<int> Nb(ntbins);   // bin counts
   for ( bin = 0; bin < ntbins; bin++) Nb[bin] = 0;
   double PI = 3.1415926535;
   double db = PI/( (double) ntbins);   // bin size
           
   for ( unsigned int iter = 0; iter < Ts; iter++){
   
      for ( int k = 0; k < d; k++){
         q[k] = chain[ k + d*iter];
      }
      ut = trans(t)*( q - cc);
      u1 = trans(n1)*( q - cc);
      u2 = trans(n2)*( q - cc);
      theta = asin(u2/(sqrt(u1*u1 + u2*u2)));
      thetas[iter] = theta;
      bin = (int) ( ( theta + .5*PI )/ db);
      Nb[bin]++;
   }        // end of analysis loop
   cout << " " << endl;
   for ( bin = 0; bin < ntbins; bin++){
      cout << " bin " << bin << " has count " << Nb[bin] << endl;
   }
   cout << " " << endl;

   
   
//   Histogram of the x coordinates q[0]=x, q[1]=y, q[2]=z
   
   int nx   = 100;             // number of x1 values for the PDF and number of x1 bins
   vector<double> Ratio(nx);   // contains the random variables Ri_N

   if ( integrate == 1 ) {
      int ni   = 500;     // number of integration points in each direction
      double L = -3.0;
      double R =  3.0;
      double x1    = .5;
      vector<double> fl(nx);  // approximate (un-normalized) true pdf for q[0]=x, compute by integrating out y,z variables
      vector<int>    Nxb(nx); // vector counting number of samples in each bin
      for ( bin = 0; bin < nx; bin++) Nxb[bin] = 0;
      double dx = ( R - L )/( (double) nx);
      for ( int i=0; i<nx; i++){
         x1 = L + dx*i + .5*dx;
         fl[i]= M.yzIntegrate( x1, L, R, eps, ni);
      }
               
      int outliers = 0;       //  number of samples outside the histogram range
      for ( unsigned int iter = 0; iter < Ts; iter++){
         x1 = chain[ d*iter ];  // same as before, but with k=0 as we need q[0] of iter-th Soft sample
         bin = (int) ( ( x1 - L )/ dx);
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
            x1 = L + dx*bin + .5*dx;
            Z = ( (double) Nxb[bin]) / ( (double) Ts*dx*fl[bin]);
            Ratio[bin] = Z;
            StringLength = snprintf( OutputString, sizeof(OutputString), " %4d   %8.3f   %8d   %9.3e    %9.3e", bin, x1, Nxb[bin], fl[bin], Z);
            cout << OutputString << endl;
         }
      }
      
      // Compute Relative Standard Error:
      
      // Define the range for the bins to be included in the computation of rel standard error
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
   StringLength = snprintf( OutputString, sizeof(OutputString),"gamma = %10.5e", gamma);
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
   OutputFile << "ModelName = \"" << M.ModelName() << "\"" << endl;
   OutputFile.close();


// Write chain in binary format (in a file called "chain.bin"), much faster for Python to read when chain is very long
   size_t sample_size = Ts*d;
   ofstream ostream1("chain.bin", ios::out | ios::binary);
   ostream1.write((char*)&chain[0], sample_size * sizeof(double));
   ostream1.close();

// Write thetas in binary format (in a file called "thetas.bin"). Contains angle theta of each sample
   size_t theta_size = Ts;
   ofstream ostream2("thetas.bin", ios::out | ios::binary);
   ostream2.write((char*)&thetas[0], theta_size * sizeof(double));
   ostream2.close();
   
}  // end of main

