#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include "tables.hpp"

template <std::size_t n, typename Function, typename T>
inline static T gauss_legendre(Function &&f, const T &a, const T &b) {
  const auto c1 = (b + a) / T(2);
  const auto c2 = (b - a) / T(2);

  const Eigen::Matrix<T, Eigen::Dynamic, 1> x =
      gauss_legendre_nodes_weights<n>().first.template cast<T>();
  const Eigen::Matrix<T, Eigen::Dynamic, 1> w =
      gauss_legendre_nodes_weights<n>().second.template cast<T>();

  return c2 * w.dot((c1 + c2 * x.array()).matrix().unaryExpr(f));
}

template <typename Function, typename T>
inline static T simpsons1_3(Function &&f, const T &a, const T &b) {
  return (b - a) * (f(a) + T(4) * f((a + b) / T(2)) + f(b)) / T(6);
}

template <typename Function, typename T>
inline static T simpsons3_8(Function &&f, const T &a, const T &b) {
  return (b - a) *
         (f(a) + T(3) * f((T(2) * a + b) / T(3)) +
          T(3) * f((a + T(2) * b) / T(3)) + f(b)) /
         T(8);
}

#endif /* QUADRATURE_HPP */
