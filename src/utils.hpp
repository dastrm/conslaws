#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>

#include <iostream>

constexpr std::string_view delim1 = "--------------------------------";
constexpr std::string_view delim2 = "================================";

inline static void print_red_endl(const std::string &str) {
  std::cout << "\033[1;31m" + str + "\033[0m" << std::endl;
}

inline static void print_green_endl(const std::string &str) {
  std::cout << "\033[92m" + str + "\033[0m" << std::endl;
}

inline static void verify(const bool &cond, const std::string &msg) {
  if (!cond) {
    print_red_endl("Error: " + msg);
    exit(EXIT_FAILURE);
  }
}

// approximate Jacobian of F at x using numerical differentiation
template <typename T, typename Function>
inline static Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
numdiff(Function &&F, const Eigen::Matrix<T, Eigen::Dynamic, 1> &x) {
  const std::size_t n = x.size();
  constexpr T h = std::sqrt(T(2) * std::numeric_limits<T>::epsilon());

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    // i-th gradient
    J.row(i) =
        F(x + h * Eigen::Matrix<T, Eigen::Dynamic, 1>::Unit(n, i)) - F(x);
  }
  J /= h;

  return J;
}

// approximate sparse Jacobian of F at x using numerical differentiation
template <typename T, typename Function>
inline static Eigen::SparseMatrix<T>
numdiff_sparse(Function &&F, const Eigen::Matrix<T, Eigen::Dynamic, 1> &x) {
  const std::size_t n = x.size();
  constexpr T h = std::sqrt(T(2) * std::numeric_limits<T>::epsilon());

  Eigen::SparseMatrix<T> J(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    // i-th gradient, unfortunately can not write to sparse row
    J.col(i) =
        (F(x + h * Eigen::Matrix<T, Eigen::Dynamic, 1>::Unit(n, i)) - F(x))
            .sparseView();
  }
  J /= h;

  return J.transpose();
}

// damped Newton based iterative (non-)linear system of equations solver
template <typename T, typename F1, typename F2>
inline static bool damped_newton(
    F1 &&F, F2 &&DF, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
    const T &rtol = 1e-4, const T &atol = 1e-6, const bool &verbose = true) {
  const std::size_t n = x.size();
  constexpr T lmin = T(1E-3); // minimal damping factor
  T lambda = T(1);            // initial and actual damping factor
  T sn, stn;                  // norms of Newton corrections
  Eigen::Matrix<T, Eigen::Dynamic, 1> s(n), st(n); // Newton corrections
  Eigen::Matrix<T, Eigen::Dynamic, 1> xn(n);       // tentative new iterate

  do {
#if 1
    // LU-factorize Jacobian with partial pivoting
    auto jacfac = DF(x).lu();
#elif 1
    // LU-factorize Jacobian with full pivoting
    auto jacfac = DF(x).fullPivLu();
#elif 1
    // QR-factorize Jacobian
    auto jacfac = DF(x).householderQr();
#else
    // SVD-factorize Jacobian
    auto jacfac = DF(x).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
    // if (jacfac.info() != Eigen::Success) {
    //   if (verbose)
    //     print_red_endl(" Jacobian factorization impossible");
    //   return false;
    // }
    s = jacfac.solve(F(x)); // Newton correction
    sn = s.norm();          // norm of Newton correction
    lambda *= T(2);
    do {
      lambda /= T(2);
      if (lambda < lmin) {
        if (verbose)
          print_red_endl(" No convergence: lambda -> 0");
        return false;
      }
      xn = x - lambda * s;      // tentative next iterate
      st = jacfac.solve(F(xn)); // simplified Newton correction
      stn = st.norm();
    } while (stn > (T(1) - lambda / T(2)) * sn); // natural monotonicity test
    x = xn;                                 // xn accepted as new iterate
    lambda = std::min(T(2) * lambda, T(1)); // try to mitigate damping
  } while ((stn > rtol * x.norm()) && (stn > atol));
  return true;
}

// damped Newton based iterative sparse (non-)linear system of equations solver
template <typename T, typename F1, typename F2>
inline static bool damped_newton_sparse(
    F1 &&F, F2 &&DF, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> x,
    const T &rtol = 1e-4, const T &atol = 1e-6, const bool &verbose = true) {
  const std::size_t n = x.size();
  constexpr T lmin = T(1E-3); // minimal damping factor
  T lambda = T(1);            // initial and actual damping factor
  T sn, stn;                  // norms of Newton corrections
  Eigen::Matrix<T, Eigen::Dynamic, 1> s(n), st(n); // Newton corrections
  Eigen::Matrix<T, Eigen::Dynamic, 1> xn(n);       // tentative new iterate
  Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;

  do {
    // LU-factorize sparse Jacobian
    solver.compute(DF(x));
    if (solver.info() != Eigen::Success) {
      if (verbose)
        print_red_endl(" LU decomposition failed");
      return false;
    }
    s = solver.solve(F(x)); // Newton correction
    if (solver.info() != Eigen::Success) {
      if (verbose)
        print_red_endl(" LU solving failed");
      return false;
    }
    sn = s.norm(); // norm of Newton correction
    lambda *= T(2);
    do {
      lambda /= T(2);
      if (lambda < lmin) {
        if (verbose)
          print_red_endl(" No convergence: lambda -> 0");
        return false;
      }
      xn = x - lambda * s;      // tentative next iterate
      st = solver.solve(F(xn)); // simplified Newton correction
      if (solver.info() != Eigen::Success) {
        if (verbose)
          print_red_endl(" LU solving failed");
        return false;
      }
      stn = st.norm();
    } while (stn > (T(1) - lambda / T(2)) * sn); // natural monotonicity test
    x = xn;                                 // xn accepted as new iterate
    lambda = std::min(T(2) * lambda, T(1)); // try to mitigate damping
  } while ((stn > rtol * x.norm()) && (stn > atol));
  return true;
}

// returns the upwind argument
template <typename T, typename F1, typename F2>
inline static T upw(F1 &&Flux, F2 &&DFlux, const T &u1, const T &u2) {
  const T s = (u1 != u2 ? (Flux(u2) - Flux(u1)) / (u2 - u1) : DFlux(u1));
  return (s >= T(0) ? u1 : u2);
}

// minmod function for slope limiter
template <typename T>
inline static T minmod(const T &u1, const T &u2) {
  if (u1 * u2 > T(0)) {
    if (std::abs(u1) <= std::abs(u2)) {
      return u1;
    }
    return u2;
  }
  return T(0);
}

// generalized minmod function for slope limiter
template <typename T, typename F1, typename F2>
inline static T gen_minmod(F1 &&Flux, F2 &&DFlux, const T &u1, const T &u2,
                           const T &h) {
  // M is an upper bound of the absolute value of |u''| at local extrema
  constexpr T M = T(1e0);
  const T u_upw = upw(Flux, DFlux, u1, u2);
  if (std::abs(u_upw) <= M * h * h) {
    return u_upw;
  }
  return minmod(u1, u2);
}

// signum function
template <typename T>
inline static int sgn(const T &val) {
  return (T(0) < val) - (val < T(0));
}

#endif /* UTILS_HPP */
