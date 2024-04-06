//
//  Model.hpp
//
//  model name: 3D warped torus
//
//  Adapted from Jonathan Goodman's foliation sampler code.
//
/*   This model defines a warped circle in 3D as the intersection
     of two surfaces.  One is a round sphere defined by
          
            | q - c_0 | = r_0
    
     The other surface is an ellipsoid, which is a distorted sphere,
     defined by
     
          sum_k ( q_k - c_{1,k} )^2/s_k^2 = 1
     
*/

#ifndef Model_hpp
#define Model_hpp

#include <iostream>
#include <blaze/Math.h>
using namespace std;
using namespace blaze;

class Model{

   public:
      double                              r;    // radius of sphere
      DynamicVector<double>               s;    // dimensional radius numbers for the ellipsoid
      DynamicMatrix<double, columnMajor>  c;    // column k = center of sphere k
      DynamicVector<double, columnVector> ssq;  // squares of the "radius" parameters of the ellipsoid
      int d;     // dimension of the ambient space
      int m;     // number of constraint functions
      int n;     // dimension of the hard constraint manifold = d-m
      DynamicVector<double, columnVector> masses; // Vector of masses
      DynamicMatrix<double, columnMajor> M; // Optional mass matrix parameter
      DynamicMatrix<double, columnMajor> M_sqrt;  // Square root of M
      DynamicMatrix<double, columnMajor> M_sqrt_inv;  // Inverse of the square root of M
      
      // Method to compute square root and inverse of square root of M, if M is initialized
      bool 
      computeMSqrtAndInv();
   
      double                                          /* evaluates the                  */
      V( DynamicVector<double, columnVector> q );     /* potential V(q)                 */
   
      DynamicVector<double, columnVector>              /* return the gradient ...       */
      gV( DynamicVector<double, columnVector> q );     /* ... of potential: grad V(q)   */
      
      DynamicVector<double, columnVector>             /* return the values ...            */
      xi( DynamicVector<double, columnVector> q );     /* ... of the constraint functions */
      
      DynamicMatrix<double, columnMajor>              /* column k is the gradient ...     */
      gxi( DynamicVector<double, columnVector> q );    /* ... of xi_k(q)                  */
   
      // Returns gxi(q) augmented to a square matrix (in case d > m), just appends columns of zeros.
      DynamicMatrix<double, columnMajor>
      Agxi(DynamicMatrix<double, columnMajor>& gxi);
   
      // Multiplies the top d-m rows of gxi by c1 and the bottom m rows by c2
      DynamicMatrix<double, columnMajor>
      scaled_gxi(const DynamicMatrix<double, columnMajor>& gxi, double c1, double c2);
   
      // DEFAULT Constructor
      Model();
   
      // Parametrized constructor with masses
      Model(int d,                                              /* dimension of the ambient space */
            int m,                                              /* number of constraints          */
            double r,                                           /* radus of sphere                              */
            const DynamicVector<double>& s,                     /* dimensional radius numbers for the ellipsoid */
            const DynamicMatrix<double, columnMajor>& c,        /* column 0 = cdnter of round sphere,  column 1 = center of ellipsoid  */
            const DynamicVector<double, columnVector>& masses); /* vector containing masses for diagonal mass tensor */
   
      string ModelName();                  /* Return the name of the model */
      
      double yzIntegrate( double x,        /* Integrate e^{-beta*U(x,y,z)} over y and z */
                          double Ly,       /* Integrate from y = Ly to y = Ry    */
                          double Ry,
                          double Lz,       /* Integrate from z = Lz to z = Rz    */
                          double Rz,
                          double eps,      /* The oscillatory parameter */
                          int niy,          /* the number of integration points in y dir */
                          int niz);         /* the number of integration points in z dir */

};

#endif /* Model.hpp */
