//
//  Model.cpp
//
//  model name: 3D warped torus
//
//  Adapted from Jonathan Goodman's foliation sampler code.
//
/*
     Enter description
*/
#include <iostream>
#include <blaze/Math.h>
#include "model.hpp"
using namespace std;
using namespace blaze;



// Default constructor
Model::Model() {}

// Parametrized constructor with masses
Model::Model(int d, int m, double D0, double a, double kappa, double lambda,
             const DynamicVector<double, columnVector>& masses)
   : d(d), m(m), D0(D0), a(a), kappa(kappa), lambda(lambda), masses(masses) {
   // Construct M as a diagonal matrix from masses and compute square root and inverse
   computeMSqrtAndInv();
}

bool Model::computeMSqrtAndInv() {
   // Ensure that the masses vector is not empty and d is set properly
   if (masses.size() > 0) {
      // Initialize M, M_sqrt, and M_sqrt_inv as d by d zero matrices
      M = DynamicMatrix<double, columnMajor>(d, d, 0.0);
      M_sqrt = DynamicMatrix<double, columnMajor>(d, d, 0.0);
      M_sqrt_inv = DynamicMatrix<double, columnMajor>(d, d, 0.0);
      
      // Compute the square root and inverse of the square root of masses for the diagonal
      for (size_t i = 0; i < d; i++) {
         double sqrtMass = sqrt(masses[i]);
         double invSqrtMass = 1.0 / sqrtMass;

         // Set the diagonal elements for M, M_sqrt, and M_sqrt_inv
         M(i,i) = masses[i];
         M_sqrt(i,i) = sqrtMass;
         M_sqrt_inv(i,i) = invSqrtMass;
      }

      return true;
   } else {
      std::cerr << "Vector 'masses' is not initialized or 'd' is not properly set." << std::endl;
      return false;
   }
}

double 
Model::V(DynamicVector<double, columnVector>& q) {
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling

   // Potential energy calculation using the transformed vector qtilde = M_sqrt_inv * q
   double V = D0 * pow(q_res[0]*q_res[0] - a*a, 2) +
            0.5 * kappa * q_res[1]*q_res[1] + lambda * q_res[0] * q_res[1];

   return V;
}


DynamicVector<double, columnVector>
Model::gV(DynamicVector<double, columnVector>& q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   
   DynamicVector<double, columnVector> gV(d);
   
   gV[0] = 4. * D0 * (q_res[0]*q_res[0] - a*a) * q_res[0] + lambda * q_res[1];
   gV[1] = kappa * q_res[1] + lambda * q_res[0];
   gV[2] = 0.;
   
   return M_sqrt_inv * gV; // mass rescaling for gradient
}

DynamicVector<double, columnVector>
Model::xi(DynamicVector<double, columnVector>& q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   DynamicVector<double, columnVector> xi(m);      // constraint function values
   xi[0] = q_res[0] - q_res[2];
   
   return xi;
}

DynamicMatrix<double, columnMajor>
Model::gxi(DynamicVector<double, columnVector>& q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   DynamicMatrix<double, columnMajor> gxi(d,m);  // gradient, (dxm) matrix
   gxi(0, 0) = 1.;
   gxi(1, 0) = 0.;
   gxi(2, 0) = -1.;
   
   return M_sqrt_inv * gxi; // mass rescaling for gradient
}

// Returns gxi(q) augmented to a square matrix (in case d > m), last n=d-m columns are just zeros.
// Needed to have a complete QR decomposition of gxi(q), with Q (dxd) matrix
DynamicMatrix<double, columnMajor>
Model::Agxi(DynamicMatrix<double, columnMajor>& gxi){
   
   DynamicMatrix<double, columnMajor> Agxi(d,d);   // Augmented gradient, (dxd) matrix
   Agxi = 0.;  // initialize to zero
   
   for (int j = 0; j < m; j++){
      column(Agxi, j)  = column(gxi, j);
   }

   return Agxi;  // last n=d-m columns are zeros
}

// Multiplies the top d-m rows of gxi by c1 and the bottom m rows by c2
DynamicMatrix<double, columnMajor>
Model::scaled_gxi(const DynamicMatrix<double, columnMajor>& gxi, double c1, double c2) {

    // Create a copy of the input matrix to hold the results
    DynamicMatrix<double, columnMajor> scaled_gxi = gxi;

    // Make sure that d is greater than m
    if(d <= m) {
        cout << " The number of rows d must be greater than m! " << endl;
        return scaled_gxi; // Return the unchanged copy in case of error
    }
   
    // Multiply the top d-m rows by c1
    auto top = submatrix(scaled_gxi, 0, 0, d-m, m);
    top *= c1;

    // Multiply the bottom m rows by c2
    auto bottom = submatrix(scaled_gxi, d-m, 0, m, m);
    bottom *= c2;

    return scaled_gxi;
}

string Model::ModelName(){                  /* Return the name of the model */
   return(" 2D Double Well with Extended Variable ");
}

//  Compute the (un-normalized) probability density for q1 by integrating over q2.
double Model::yzIntegrate(double x, double Ly, double Ry, double Lz, double Rz, double eps, int niy, int niz){

//  Use the rectangle rule in the y and z directions with n midpoints in each direction

   double dy  = (Ry-Ly)/( (double) niy);
   double dz  = (Rz-Lz)/( (double) niz);
   double y, z;
   double sum = 0.;
   
   DynamicVector<double, columnVector> qv( d);    // point in 3D
   DynamicVector<double, columnVector> xiv( m);   // values of the constraint functions
   double Vqv( d);                                // values of potential
   
   qv[0] = x;
   for ( int j = 0; j < niy; j++){
      y = Ly + j*dy + .5*dy;
      qv[1] = y;
      for ( int k = 0; k < niz; k++) {
         z = Lz + k*dz + .5*dz;
         qv[2] = z;
         qv  = M_sqrt * qv;   // since xi(qv) = \xi( M_sqrt_inv * qv ) and V(qv) = \V( M_sqrt_inv * qv )
         xiv = xi(qv);
         Vqv = V(qv);
         sum += exp( - Vqv - 0.5*(trans(xiv)*xiv)/(eps*eps) );
      }
   }
   return dz*dy*sum;
}
