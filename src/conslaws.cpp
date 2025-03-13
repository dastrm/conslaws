#include "conslaws.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using data_t = float;

void plot_solution() {
  const auto t0_glob = std::chrono::high_resolution_clock::now();

  bool enable;
  std::size_t rhs_num, rk_num, cells;
  data_t x_min, x_max, cfl_number;
  std::vector<data_t> t_ends;

  {
    std::ifstream fin("../config/plot_solution.ini");
    std::string line;
    int old_idx;

    std::getline(fin, line);
    enable = std::stoi(line);
    if (!enable)
      return;

    std::getline(fin, line);
    rhs_num = std::stoul(line);

    std::getline(fin, line);
    rk_num = std::stoul(line);

    std::getline(fin, line);
    x_min = std::stod(line);

    std::getline(fin, line);
    x_max = std::stod(line);

    std::getline(fin, line);
    old_idx = -1;
    do {
      const int idx = line.find_first_of(",", old_idx + 1);
      t_ends.push_back(std::stod(line.substr(old_idx + 1, idx)));
      old_idx = idx;
    } while (static_cast<std::size_t>(old_idx) < line.size());
    std::sort(t_ends.begin(), t_ends.end());

    std::getline(fin, line);
    cfl_number = std::stod(line);

    std::getline(fin, line);
    cells = std::stoul(line);

    fin.close();
  }

  std::cout << delim2 << std::endl;

  constexpr auto Flux = [](const data_t &u) { return u; };
  constexpr auto DFlux = [](const data_t & /*u*/) { return data_t(1.0); };
  static const auto ICs = [x_min, x_max](const data_t &x) {
    // return x * x;
    // return x < (x_max + x_min) / 2 ? 1.0 : 0.0;
    return std::sin(2 * data_t(M_PI) * x / (x_max - x_min));
  };

  Eigen::Matrix<data_t, Eigen::Dynamic, 1> xc, mu;
  data_t t_end_old = 0;
  for (const auto &t_end : t_ends) {
    const std::string ofstring =
        "solution_" + std::format("{}", t_end) + ".csv";
    std::ofstream of(ofstring);
    of << "xc,mu" << std::setprecision(10) << std::scientific << std::endl;

    const auto sol = solve(cells, x_min, x_max, t_end - t_end_old, cfl_number,
                           Flux, DFlux, ICs, rhs_num, rk_num, xc, mu);
    xc = sol.first;
    mu = sol.second;

    for (std::size_t i = 0; i < cells; ++i) {
      of << xc(i) << "," << mu(i) << std::endl;
    }

    of.close();
    std::cout << "Saved " + ofstring << std::endl;

    t_end_old = t_end;
  }

  const auto t1_glob = std::chrono::high_resolution_clock::now();
  std::cout << delim1 << std::endl;
  std::cout << "Done in "
            << std::chrono::duration<data_t>(t1_glob - t0_glob).count() << "s"
            << std::endl;
  std::cout << delim2 << std::endl;
}

void plot_efficiency() {
  const auto t0_glob = std::chrono::high_resolution_clock::now();

  bool enable;
  data_t warmup_time, min_runtime, x_min, x_max, t_end;
  std::vector<std::size_t> rhs_nums, rk_nums;
  std::vector<data_t> cfl_numbers;

  {
    std::ifstream fin("../config/plot_efficiency.ini");
    std::string line;
    int old_idx;

    std::getline(fin, line);
    enable = std::stoi(line);
    if (!enable)
      return;

    std::getline(fin, line);
    warmup_time = std::stod(line);

    std::getline(fin, line);
    min_runtime = std::stod(line);

    std::getline(fin, line);
    old_idx = -1;
    do {
      const int idx = line.find_first_of(",", old_idx + 1);
      rhs_nums.push_back(std::stoi(line.substr(old_idx + 1, idx)));
      old_idx = idx;
    } while (static_cast<std::size_t>(old_idx) < line.size());

    std::getline(fin, line);
    old_idx = -1;
    do {
      const int idx = line.find_first_of(",", old_idx + 1);
      rk_nums.push_back(std::stoi(line.substr(old_idx + 1, idx)));
      old_idx = idx;
    } while (static_cast<std::size_t>(old_idx) < line.size());

    std::getline(fin, line);
    x_min = std::stod(line);

    std::getline(fin, line);
    x_max = std::stod(line);

    std::getline(fin, line);
    t_end = std::stod(line);

    std::getline(fin, line);
    old_idx = -1;
    do {
      const int idx = line.find_first_of(",", old_idx + 1);
      cfl_numbers.push_back(std::stod(line.substr(old_idx + 1, idx)));
      old_idx = idx;
    } while (static_cast<std::size_t>(old_idx) < line.size());

    fin.close();
  }

  std::cout << delim2 << std::endl;

  constexpr auto Flux = [](const data_t &u) { return u; };
  constexpr auto DFlux = [](const data_t & /*u*/) { return data_t(1.0); };
  static const auto ICs = [x_min, x_max](const data_t &x) {
    // return x * x;
    return x < (x_max + x_min) / 2 ? data_t(1.0) : data_t(0.0);
    return std::sin(2 * data_t(M_PI) * x / (x_max - x_min));
  };
  static const auto Exact = [&](const data_t &x) { return ICs(x); };

  const auto num_pairs = rhs_nums.size();
  // loop over pairs of rhs_num and rk_nums
  for (std::size_t i = 0; i < num_pairs; ++i) {
    const auto rhs_num = rhs_nums[i];
    const auto rk_num = rk_nums[i];
    for (const auto &cfl_number : cfl_numbers) {
      const std::string ofstring = "efficiency_" +
                                   std::format("{:02}", rhs_num) + "_" +
                                   std::format("{:02}", rk_num) + "_" +
                                   std::format("{}", cfl_number) + ".csv";
      std::ofstream of(ofstring);
      of << "runtime,L1" << std::setprecision(10) << std::scientific
         << std::endl;
      for (std::size_t n = 0, runs = 0; runs != 1; ++n) {
        runs = 0;
        const std::size_t cells = 10 * std::pow(data_t(1.75), n);
        const data_t h = (x_max - x_min) / cells;
        Eigen::Matrix<data_t, Eigen::Dynamic, 1> xc, mu;

        // warmup
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = t0;
        do {
          const auto sol = solve(cells, x_min, x_max, data_t(1e-6), cfl_number,
                                 Flux, DFlux, ICs, rhs_num, rk_num);
          xc = sol.first;
          mu = sol.second;
          t1 = std::chrono::high_resolution_clock::now();
        } while (std::chrono::duration<data_t>(t1 - t0).count() < warmup_time);

        // measure
        data_t runtime = 0;
        do {
          t0 = std::chrono::high_resolution_clock::now();
          const auto sol = solve(cells, x_min, x_max, t_end, cfl_number, Flux,
                                 DFlux, ICs, rhs_num, rk_num);
          xc = sol.first;
          mu = sol.second;
          t1 = std::chrono::high_resolution_clock::now();
          ++runs;
          runtime += std::chrono::duration<data_t>(t1 - t0).count();
        } while (runtime < min_runtime);
        runtime /= runs;

        const data_t L1 =
            get_L1_error(cells, h, xc, mu, Exact, Flux, DFlux, rhs_num);
        of << runtime << "," << L1 << std::endl;
      }

      of.close();
      std::cout << "Saved " + ofstring << std::endl;
    }
  }

  const auto t1_glob = std::chrono::high_resolution_clock::now();
  std::cout << delim1 << std::endl;
  std::cout << "Done in "
            << std::chrono::duration<data_t>(t1_glob - t0_glob).count() << "s"
            << std::endl;
  std::cout << delim2 << std::endl;
}

int main() {
  plot_solution();
  plot_efficiency();
  return 0;
}
