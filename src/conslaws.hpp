#ifndef CONSLAWS_HPP
#define CONSLAWS_HPP

#include "quadrature.hpp"
#include "integrators.hpp"

template <typename T, typename F1, typename F2, typename F3>
inline static T get_L1_error(const std::size_t &cells, const T &h,
                             const Eigen::Matrix<T, Eigen::Dynamic, 1> &xc,
                             const Eigen::Matrix<T, Eigen::Dynamic, 1> &mu,
                             F1 &&Ref, F2 &&Flux, F3 &&DFlux,
                             const std::size_t &rhs_num) {
  T L1_enum = 0, L1_denom = 0;
  for (std::size_t i = 0; i < cells; ++i) {
    const std::size_t im1 = (i == 0 ? cells - 1 : i - 1);
    const std::size_t ip1 = (i == cells - 1 ? 0 : i + 1);

    static const auto L1_enum_func = [&](const T &x) {
      T sol = mu(i);
      if (rhs_num == 2)
        sol += upw(Flux, DFlux, mu(ip1) - mu(i), mu(i) - mu(im1)) *
               (x - xc(i)) / h;
      else if (rhs_num == 3)
        sol += gen_minmod(Flux, DFlux, mu(ip1) - mu(i), mu(i) - mu(im1), h) *
               (x - xc(i)) / h;
      return std::abs(Ref(x) - sol);
    };
    static const auto L1_denom_func = [&](const T &x) {
      return std::abs(Ref(x));
    };

    /*
    L1_enum += L1_enum_func(xc(i));
    L1_denom += L1_denom_func(xc(i));
    */
    L1_enum +=
        gauss_legendre<4>(L1_enum_func, xc(i) - h / T(2), xc(i) + h / T(2));
    L1_denom +=
        gauss_legendre<4>(L1_denom_func, xc(i) - h / T(2), xc(i) + h / T(2));
  }
  return L1_enum / L1_denom;
}

template <typename T, typename F1, typename F2, typename F3>
inline static T get_Linf_error(const std::size_t &cells, const T &h,
                               const Eigen::Matrix<T, Eigen::Dynamic, 1> &xc,
                               const Eigen::Matrix<T, Eigen::Dynamic, 1> &mu,
                               F1 &&Ref, F2 &&Flux, F3 &&DFlux,
                               const std::size_t &rhs_num) {
  T Linf_enum = 0, Linf_denom = 0;
  for (std::size_t i = 0; i < cells; ++i) {
    const std::size_t im1 = (i == 0 ? cells - 1 : i - 1);
    const std::size_t ip1 = (i == cells - 1 ? 0 : i + 1);

    static const auto Linf_enum_func = [&](const T &x) {
      T sol = mu(i);
      if (rhs_num == 2)
        sol += upw(Flux, DFlux, mu(ip1) - mu(i), mu(i) - mu(im1)) *
               (x - xc(i)) / h;
      else if (rhs_num == 3)
        sol += gen_minmod(Flux, DFlux, mu(ip1) - mu(i), mu(i) - mu(im1), h) *
               (x - xc(i)) / h;
      return std::abs(Ref(x) - sol);
    };
    static const auto Linf_denom_func = [&](const T &x) {
      return std::abs(Ref(x));
    };

    /*
    Linf_enum += Linf_enum_func(xc(i));
    Linf_denom += Linf_denom_func(xc(i));
    */
    Linf_enum =
        std::max(Linf_enum, gauss_legendre<4>(Linf_enum_func, xc(i) - h / T(2),
                                              xc(i) + h / T(2)));
    Linf_denom = std::max(
        Linf_denom,
        gauss_legendre<4>(Linf_denom_func, xc(i) - h / T(2), xc(i) + h / T(2)));
  }
  return Linf_enum / Linf_denom;
}

template <typename T, typename F1, typename F2, typename F3>
inline static auto solve(const std::size_t &cells, const T &x_min,
                         const T &x_max, const T &t_end, const T &cfl_number,
                         F1 &&Flux, F2 &&DFlux, F3 &&ICs,
                         const std::size_t &rhs_num, const std::size_t &rk_num,
                         const Eigen::Matrix<T, Eigen::Dynamic, 1> &xc_init =
                             Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(0),
                         const Eigen::Matrix<T, Eigen::Dynamic, 1> &mu_init =
                             Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(0)) {
  const T h = (x_max - x_min) / cells;
  const T cfl_times_h = cfl_number * h;

  Eigen::Matrix<T, Eigen::Dynamic, 1> xc(cells), mu(cells);
  if (xc_init.size() != 0 || mu_init.size() != 0) {
    verify(static_cast<std::size_t>(xc_init.size()) == cells,
           "Inconsistent solve args.");
    verify(static_cast<std::size_t>(mu_init.size()) == cells,
           "Inconsistent solve args.");

    xc = xc_init;
    mu = mu_init;
  } else {
    xc = Eigen::Matrix<T, Eigen::Dynamic, 1>::LinSpaced(cells, x_min + h / T(2),
                                                        x_max - h / T(2));
#if 1 // init cell averages with numerical quadrature
    for (std::size_t i = 0; i < cells; ++i) {
      mu(i) = gauss_legendre<4>(ICs, xc(i) - h / T(2), xc(i) + h / T(2)) / h;
    }
#else
    mu = xc.unaryExpr(ICs);
#endif
  }

#if 0
  // local Lax-Friedrichs numerical flux
  static const auto NFlux = [&](const T &u1, const T &u2) {
    const T s_max = std::max(std::abs(DFlux(u1)), std::abs(DFlux(u2)));
    return (Flux(u1) + Flux(u2) - s_max * (u2 - u1)) / T(2);
  };
#elif 0
  // more accurate local Lax-Friedrichs numerical flux
  static const auto NFlux = [&](const T &u1, const T &u2) {
    const T s_max =
        std::max({std::abs(DFlux(u1)), std::abs(DFlux((u1 + u2) / T(2))),
                  std::abs(DFlux(u2))});
    return (Flux(u1) + Flux(u2) - s_max * (u2 - u1)) / T(2);
  };
#elif 0
  // more accurate Godunov numerical flux
  static const auto NFlux = [&](const T &u1, const T &u2) {
    if (u1 < u2)
      return std::min({Flux(u1), Flux((u1 + u2) / T(2)), Flux(u2)});
    return std::max({Flux(u2), Flux((u2 + u1) / T(2)), Flux(u1)});
  };
#else
  // Godunov numerical flux
  static const auto NFlux = [&](const T &u1, const T &u2) {
    if (u1 < u2)
      return std::min(Flux(u1), Flux(u2));
    return std::max(Flux(u2), Flux(u1));
  };
  // partial derivative of NFlux w.r.t. u1
  static const auto DNFlux1 = [&](const T &u1, const T &u2) {
    static const auto Dminmax = [&](const T &f1, const T &f2) {
      if (f2 >= f1)
        return DFlux(u1);
      return T(0);
    };

    if (u1 < u2)
      return Dminmax(Flux(u1), Flux(u2));
    return Dminmax(Flux(u2), Flux(u1));
  };
  // partial derivative of NFlux w.r.t. u2
  static const auto DNFlux2 = [&](const T &u1, const T &u2) {
    return -DNFlux1(u2, u1);
  };
#endif

  static const auto rhs1 = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &u) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> g(cells);

    if (cells == 1) {
      g(0) = T(0);
    } else {
      g(0) = NFlux(u(cells - 1), u(0)) - NFlux(u(0), u(1));
      if (cells > 2) {
#if 0
        for (std::size_t i = 1; i < cells - 1; ++i) {
          g(i) = NFlux(u(i - 1), u(i)) - NFlux(u(i), u(i + 1));
        }
#else
        g.segment(1, cells - 2) =
            u.segment(0, cells - 2).binaryExpr(u.segment(1, cells - 2), NFlux) -
            u.segment(1, cells - 2).binaryExpr(u.segment(2, cells - 2), NFlux);
#endif
      }
      g(cells - 1) =
          NFlux(u(cells - 2), u(cells - 1)) - NFlux(u(cells - 1), u(0));
    }

    g /= h;
    return g;
  };

  static const auto Drhs1 = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &u) {
    const std::size_t d = u.size();
    Eigen::SparseMatrix<T> J(d, d);
    J.reserve(std::min(3 * d, d * d));

    if (d == 1) {
      J.insert(0, 0) = T(0);
    } else {
      J.insert(0, d - 1) = DNFlux1(u(d - 1), u(0));
      J.insert(0, 0) = DNFlux2(u(d - 1), u(0)) - DNFlux1(u(0), u(1));
      if (d > 2) {
        J.insert(0, 1) = -DNFlux2(u(0), u(1));
        // loop through gradients
        for (std::size_t i = 1; i < d - 1; ++i) {
          J.insert(i, i - 1) = DNFlux1(u(i - 1), u(i));
          J.insert(i, i) = DNFlux2(u(i - 1), u(i)) - DNFlux1(u(i), u(i + 1));
          J.insert(i, i + 1) = -DNFlux2(u(i), u(i + 1));
        }
        J.insert(d - 1, d - 2) = DNFlux1(u(d - 2), u(d - 1));
      }
      J.insert(d - 1, d - 1) =
          DNFlux2(u(d - 2), u(d - 1)) - DNFlux1(u(d - 1), u(0));
      J.insert(d - 1, 0) = -DNFlux2(u(d - 1), u(0));
    }

    J /= h;
    return J;
  };

  static const auto rhs2 = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &u) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> g(cells);

    // linear reconstruction based on finite differences
    if (cells == 1) {
      g(0) = 0;
    } else if (cells <= 4) {
      g(0) = NFlux(u(cells - 1), u(0)) - NFlux(u(0), u(1));
      for (std::size_t i = 1; i < cells - 1; ++i) {
        g(i) = NFlux(u(i - 1), u(i)) - NFlux(u(i), u(i + 1));
      }
      g(cells - 1) =
          NFlux(u(cells - 2), u(cells - 1)) - NFlux(u(cells - 1), u(0));
    } else {
      // clang-format off
      g(0) = NFlux(u(cells-1) + upw(Flux, DFlux, u(0) - u(cells-1), u(cells-1) - u(cells-2)) / T(2),
                   u(0)       - upw(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1)) / T(2)) -
             NFlux(u(0)       + upw(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1)) / T(2),
                   u(1)       - upw(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      ) / T(2));

      g(1) = NFlux(u(0)       + upw(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1)) / T(2),
                   u(1)       - upw(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      ) / T(2)) -
             NFlux(u(1)       + upw(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      ) / T(2),
                   u(2)       - upw(Flux, DFlux, u(3) - u(2)      , u(2)       - u(1)      ) / T(2));
      for (std::size_t i = 2; i < cells - 2; ++i) {
        g(i) = NFlux(u(i-1) + upw(Flux, DFlux, u(i)   - u(i-1), u(i-1) - u(i-2)) / T(2),
                     u(i)   - upw(Flux, DFlux, u(i+1) - u(i)  , u(i)   - u(i-1)) / T(2)) -
               NFlux(u(i)   + upw(Flux, DFlux, u(i+1) - u(i)  , u(i)   - u(i-1)) / T(2),
                     u(i+1) - upw(Flux, DFlux, u(i+2) - u(i+1), u(i+1) - u(i)  ) / T(2));
      }
      g(cells-2) = NFlux(u(cells-3) + upw(Flux, DFlux, u(cells-2) - u(cells-3), u(cells-3) - u(cells-4)) / T(2),
                         u(cells-2) - upw(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3)) / T(2)) -
                   NFlux(u(cells-2) + upw(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3)) / T(2),
                         u(cells-1) - upw(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2)) / T(2));

      g(cells-1) = NFlux(u(cells-2) + upw(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3)) / T(2),
                         u(cells-1) - upw(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2)) / T(2)) -
                   NFlux(u(cells-1) + upw(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2)) / T(2),
                         u(0)       - upw(Flux, DFlux, u(1)       - u(0)      , u(0)       - u(cells-1)) / T(2));
      // clang-format on
    }

    g /= h;
    return g;
  };

  static const auto rhs3 = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1> &u) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> g(cells);

    // MUSCL scheme
    if (cells == 1) {
      g(0) = 0;
    } else if (cells <= 4) {
      g(0) = NFlux(u(cells - 1), u(0)) - NFlux(u(0), u(1));
      for (std::size_t i = 1; i < cells - 1; ++i) {
        g(i) = NFlux(u(i - 1), u(i)) - NFlux(u(i), u(i + 1));
      }
      g(cells - 1) =
          NFlux(u(cells - 2), u(cells - 1)) - NFlux(u(cells - 1), u(0));
    } else {
      // clang-format off
      g(0) = NFlux(u(cells-1) + gen_minmod(Flux, DFlux, u(0) - u(cells-1), u(cells-1) - u(cells-2), h) / T(2),
                   u(0)       - gen_minmod(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1), h) / T(2)) -
             NFlux(u(0)       + gen_minmod(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1), h) / T(2),
                   u(1)       - gen_minmod(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      , h) / T(2));

      g(1) = NFlux(u(0)       + gen_minmod(Flux, DFlux, u(1) - u(0)      , u(0)       - u(cells-1), h) / T(2),
                   u(1)       - gen_minmod(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      , h) / T(2)) -
             NFlux(u(1)       + gen_minmod(Flux, DFlux, u(2) - u(1)      , u(1)       - u(0)      , h) / T(2),
                   u(2)       - gen_minmod(Flux, DFlux, u(3) - u(2)      , u(2)       - u(1)      , h) / T(2));
      for (std::size_t i = 2; i < cells - 2; ++i) {
        g(i) = NFlux(u(i-1) + gen_minmod(Flux, DFlux, u(i)   - u(i-1), u(i-1) - u(i-2), h) / T(2),
                     u(i)   - gen_minmod(Flux, DFlux, u(i+1) - u(i)  , u(i)   - u(i-1), h) / T(2)) -
               NFlux(u(i)   + gen_minmod(Flux, DFlux, u(i+1) - u(i)  , u(i)   - u(i-1), h) / T(2),
                     u(i+1) - gen_minmod(Flux, DFlux, u(i+2) - u(i+1), u(i+1) - u(i)  , h) / T(2));
      }
      g(cells-2) = NFlux(u(cells-3) + gen_minmod(Flux, DFlux, u(cells-2) - u(cells-3), u(cells-3) - u(cells-4), h) / T(2),
                         u(cells-2) - gen_minmod(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3), h) / T(2)) -
                   NFlux(u(cells-2) + gen_minmod(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3), h) / T(2),
                         u(cells-1) - gen_minmod(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2), h) / T(2));

      g(cells-1) = NFlux(u(cells-2) + gen_minmod(Flux, DFlux, u(cells-1) - u(cells-2), u(cells-2) - u(cells-3), h) / T(2),
                         u(cells-1) - gen_minmod(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2), h) / T(2)) -
                   NFlux(u(cells-1) + gen_minmod(Flux, DFlux, u(0)       - u(cells-1), u(cells-1) - u(cells-2), h) / T(2),
                         u(0)       - gen_minmod(Flux, DFlux, u(1)       - u(0)      , u(0)       - u(cells-1), h) / T(2));
      // clang-format on
    }

    g /= h;
    return g;
  };

  static const ExplRKIntegrator ExplEuler(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(1, 1) << 0).finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(1) << 1).finished());

  static const ExplRKIntegrator HeunsMethod(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << 0, 0, 1, 0)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(2) << 1. / 2, 1. / 2).finished());

  static const ExplRKIntegrator RK3SSP(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(3, 3) << 0, 0, 0, 1, 0,
       0, 1. / 4, 1. / 4, 0)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(3) << 1. / 6, 1. / 6, 2. / 3)
          .finished());

  static const ExplRKIntegrator Ralston(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(4, 4) << 0, 0, 0, 0,
       0.4, 0, 0, 0, 0.29697761, 0.15875964, 0, 0, 0.21810040, -3.05096516,
       3.83286476, 0)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(4) << 0.17476028, -0.55148066,
       1.20553560, 0.17118478)
          .finished());

  static constexpr T ggr = (1 + 1 / std::sqrt(3)) / 2;
  static const SemiImplRKIntegrator ROW3(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << 0, 0, 2. / 3,
       0)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(2) << 1. / 4, 3. / 4).finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << ggr, 0,
       -4 * ggr / 3, ggr)
          .finished());

  static const ImplRKIntegrator ImplEuler(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(1, 1) << 1).finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(1) << 1).finished());

  static const ImplRKIntegrator ImplMidpoint(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(1, 1) << 0.5)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(1) << 1).finished());

  static const ImplRKIntegrator Radau3(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << 5. / 12,
       -1. / 12, 0.75, 0.25)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(2) << 0.75, 0.25).finished());

  static const ImplRKIntegrator Radau5(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(3, 3)
           << (88 - 7 * std::sqrt(6)) / 360,
       (296 - 169 * std::sqrt(6)) / 1800, (-2 + 3 * std::sqrt(6)) / 225,
       (296 + 169 * std::sqrt(6)) / 1800, (88 - 7 * std::sqrt(6)) / 360,
       (-2 - 3 * std::sqrt(6)) / 225, (16 - std::sqrt(6)) / 36,
       (16 + std::sqrt(6)) / 36, 1. / 9)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(3) << (16 - std::sqrt(6)) / 36,
       (16 + std::sqrt(6)) / 36, 1. / 9)
          .finished());

  static constexpr T lbd = 1 - std::sqrt(2) / 2;
  static const DiagImplRKIntegrator SDIRK3(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << lbd, 0,
       1 - lbd, lbd)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(2) << 1 - lbd, lbd).finished());
  static const ImplRKIntegrator SDIRK3_(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(2, 2) << lbd, 0,
       1 - lbd, lbd)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(2) << 1 - lbd, lbd).finished());

  static constexpr T alp = 2 * std::cos(M_PI / 18) / std::sqrt(3);
  static const DiagImplRKIntegrator SDIRK4(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(3, 3) << (1 + alp) / 2,
       0, 0, -alp / 2, (1 + alp) / 2, 0, 1 + alp, -(1 + 2 * alp), (1 + alp) / 2)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(3) << 1 / (6 * alp * alp),
       1 - 1 / (3 * alp * alp), 1 / (6 * alp * alp))
          .finished());
  static const ImplRKIntegrator SDIRK4_(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(3, 3) << (1 + alp) / 2,
       0, 0, -alp / 2, (1 + alp) / 2, 0, 1 + alp, -(1 + 2 * alp), (1 + alp) / 2)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(3) << 1 / (6 * alp * alp),
       1 - 1 / (3 * alp * alp), 1 / (6 * alp * alp))
          .finished());

  static const DiagImplRKIntegrator Lstab3(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(4, 4) << 0.5, 0, 0, 0,
       1. / 6, 0.5, 0, 0, -0.5, 0.5, 0.5, 0, 1.5, -1.5, 0.5, 0.5)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(4) << 1.5, -1.5, 0.5, 0.5)
          .finished());
  static const ImplRKIntegrator Lstab3_(
      (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(4, 4) << 0.5, 0, 0, 0,
       1. / 6, 0.5, 0, 0, -0.5, 0.5, 0.5, 0, 1.5, -1.5, 0.5, 0.5)
          .finished(),
      (Eigen::Matrix<T, Eigen::Dynamic, 1>(4) << 1.5, -1.5, 0.5, 0.5)
          .finished());

  // clang-format off
#define ODE_SOLVE(integrator)                                                   \
  if (rhs_num == 1)                                                             \
    mu = integrator.solve(rhs1, Drhs1, rhs_num, mu, t_end, cfl_times_h, DFlux); \
  if (rhs_num == 2)                                                             \
    mu = integrator.solve(rhs2, Drhs1, rhs_num, mu, t_end, cfl_times_h, DFlux); \
  if (rhs_num == 3)                                                             \
    mu = integrator.solve(rhs3, Drhs1, rhs_num, mu, t_end, cfl_times_h, DFlux);
  // clang-format on

  if (rk_num == 1) {
    ODE_SOLVE(ExplEuler)
  } else if (rk_num == 2) {
    ODE_SOLVE(HeunsMethod)
  } else if (rk_num == 3) {
    ODE_SOLVE(RK3SSP)
  } else if (rk_num == 4) {
    ODE_SOLVE(Ralston)
  } else if (rk_num == 5) {
    ODE_SOLVE(ROW3)
  } else if (rk_num == 6) {
    ODE_SOLVE(ImplEuler)
  } else if (rk_num == 7) {
    ODE_SOLVE(ImplMidpoint)
  } else if (rk_num == 8) {
    ODE_SOLVE(Radau3)
  } else if (rk_num == 9) {
    ODE_SOLVE(Radau5)
  } else if (rk_num == 10) {
    ODE_SOLVE(SDIRK3)
  } else if (rk_num == 11) {
    ODE_SOLVE(SDIRK3_)
  } else if (rk_num == 12) {
    ODE_SOLVE(SDIRK4)
  } else if (rk_num == 13) {
    ODE_SOLVE(SDIRK4_)
  } else if (rk_num == 14) {
    ODE_SOLVE(Lstab3)
  } else if (rk_num == 15) {
    ODE_SOLVE(Lstab3_)
  }

  return std::pair(xc, mu);
}

#endif /* CONSLAWS_HPP */
