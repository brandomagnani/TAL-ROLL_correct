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
Model::Model(int d,                                             /* dimension of the ambient space */
             int m,                                             /* number of constraints          */
             double r,                                          /* radus of sphere                              */
             const DynamicVector<double>& s,                    /* dimensional radius numbers for the ellipsoid */
             const DynamicMatrix<double, columnMajor>& c,       /* column 0 = cdnter of round sphere,  column 1 = center of ellipsoid  */
             const DynamicVector<double, columnVector>& masses) /* vector containing masses for diagonal mass tensor */
   : d(d), m(m), r(r), s(s), c(c), masses(masses) {
      
   // Construct M as a diagonal matrix from masses and compute square root and inverse
   computeMSqrtAndInv();
      
   ssq.resize(d); // contains the square of entries of s, the "radius" parameters of the ellipsoid
   for ( int i = 0; i < d; i++) {
      ssq[i] = s[i]*s[i];
   }
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
Model::V(DynamicVector<double, columnVector> q) {
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   return 0.5 * sqrNorm( q_res );
}

DynamicVector<double, columnVector>
Model::gV(DynamicVector<double, columnVector> q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   DynamicVector<double, columnVector> gV(d);
   
   for ( int j = 0; j < d; j++ ){
      gV[j] = q_res[j];
   }
   return M_sqrt_inv * gV; // mass rescaling for gradient
}

DynamicVector<double, columnVector>
Model::xi(DynamicVector<double, columnVector> q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   DynamicVector<double, columnVector> disp(d);   // vector from q to c_j (center of sphere j)
   DynamicVector<double, columnVector> xi(m);      // constraint function values
   
   //        The sphere
     
   disp = q_res - column(c,0);
   xi[0] = trans(disp)*disp - r*r;
   
   //       The ellipsoid

   double exi = 0.;              // xi[1] = exi = xi "for the ellipsoid"
   disp = q_res - column(c,1);
   for ( int j =0; j < d; j++){
      exi += disp[j]*disp[j]/ssq[j];
   }
   xi[1] = exi - 1.;
   
   return xi;
}

DynamicMatrix<double, columnMajor>
Model::gxi(DynamicVector<double, columnVector> q){
   
   DynamicVector<double, columnVector> q_res = M_sqrt_inv * q; // mass rescaling
   DynamicVector<double, columnVector> disp(d);   // displacement from a center
   DynamicMatrix<double, columnMajor> gxi(d,m);  // gradient, (dxm) matrix
   
   //        The sphere, first column
     
   disp = q_res - column(c,0);
   column(gxi, 0)  = 2.*disp;
   
   //       The ellipsoid, second column

   disp = q_res - column(c,1);
   for ( int j =0; j < d; j++){
      gxi(j,1) = 2.0*disp[j]/ssq[j];
   }
   
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
   return(" 3D Warped Torus with V(q) = |q|^2/2 ");
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
