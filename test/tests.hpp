#ifndef TESTS_HPP
#define TESTS_HPP

#include "gtest/gtest.h"
#include "../src/conslaws.hpp"

template <typename F1, typename F2, typename F3, typename F4>
static void test_convergence(const std::vector<std::size_t> &rhs_nums,
                             const std::vector<std::size_t> &rk_nums,
                             const std::vector<double> &order_mins,
                             const std::size_t &n_min, const std::size_t &n_max,
                             const double &x_min, const double &x_max,
                             const double &t_end, const double &cfl_number,
                             F1 &&ICs, F2 &&Exact, F3 &&Flux, F4 &&DFlux) {
  const auto num_pairs = rhs_nums.size();
  EXPECT_TRUE(num_pairs == rk_nums.size());
  EXPECT_TRUE(num_pairs == order_mins.size());

  // loop over all num_pairs
  for (std::size_t i = 0; i < num_pairs; ++i) {
    const auto rhs_num = rhs_nums[i];
    const auto rk_num = rk_nums[i];
    const auto order_min = order_mins[i];
    double L1_old = 0, Linf_old = 0, h_old = x_max - x_min;
    for (std::size_t n = n_min; n < n_max; ++n) {
      const std::size_t cells = std::pow(2, n);
      const double h = (x_max - x_min) / cells;

      const auto [xc, mu] = solve(cells, x_min, x_max, t_end, cfl_number, Flux,
                                  DFlux, ICs, rhs_num, rk_num);
      EXPECT_TRUE(xc.allFinite());
      EXPECT_TRUE(mu.allFinite());

      const double L1 =
          get_L1_error(cells, h, xc, mu, Exact, Flux, DFlux, rhs_num);
      const double Linf =
          get_Linf_error(cells, h, xc, mu, Exact, Flux, DFlux, rhs_num);
      EXPECT_TRUE(std::isfinite(L1) && L1 >= 0);
      EXPECT_TRUE(std::isfinite(Linf) && Linf >= 0);

      // check whether error is decreasing
      if (cells >= 2 << 3) {
        EXPECT_GT(L1_old / L1, 1.0);
        EXPECT_GT(Linf_old / Linf, 1.0);
      }

      // check observed order of convergence
      if (n == n_max - 1) {
        EXPECT_GT(std::log(L1_old / L1) / std::log(h_old / h), order_min);
        EXPECT_GT(std::log(Linf_old / Linf) / std::log(h_old / h), order_min);
      }

      L1_old = L1;
      Linf_old = Linf;
      h_old = h;
    }
  }
}

#endif /* TESTS_HPP */
