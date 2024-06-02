import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

# Sistem Evaluasi Supplier dengan Integrasi Metode BWM dan Fuzzy MARCOS

# Input
# Perbandingan Kriteria terbaik dengan Semua Kriteria (Best-to-Others)
bto = np.array([1, 2, 3, 8, 7, 9])

# Perbandingan Semua Kriteria dengan Kriteria Terburuk (Others-to-Worst)
otw = np.array([9, 8, 6, 2, 3, 1])

# Penentuan Jenis Kriteria
criterion_type = ["max", "max", "max", "max", "max", "max"]

#Dataset lower (Misal TFN = 5,7,9 maka yang diambil angka 5)
dataset_l = np.array([
  [ (6), (7), (7), (6), (7), (7) ], #a1
  [ (5), (7), (6), (4), (7), (5) ], #a2
  [ (5), (7), (5), (4), (6), (5) ], #a3
  [ (7), (7), (5), (4), (7), (6) ] #a4
])

#Dataset middle (Misal TFN = 5,7,9 maka yang diambil angka 7)
dataset_m = np.array([
  [ (8), (9), (9), (8), (9), (9) ], #a1
  [ (7), (9), (8), (6), (9), (7) ], #a2
  [ (7), (9), (7), (6), (8), (7) ], #a3
  [ (9), (9), (7), (6), (9), (8) ] #a4
])

#Dataset upper (Misal TFN = 5,7,9 maka yang diambil angka 9)
dataset_u = np.array([
  [ (9), (9), (9), (9), (9), (9) ], #a1
  [ (9), (9), (9), (8), (9), (9) ], #a2
  [ (9), (9), (9), (8), (9), (9) ], #a3
  [ (9), (9), (9), (8), (9), (9) ] #a4
])

# Perhitungan
# Integrasi Metode BWM dengan Fuzzy MARCOS

def best_worst_method(bto, otw, eps_penalty = 1, verbose = True):
    cr = []
    mx = np.max(bto)

    def target_function(variables):
        eps     = variables[-1]
        wx      = variables[np.argmin(bto)]
        wy      = variables[np.argmin(otw)]
        cons_1  = []
        cons_2  = []
        penalty = 0
        for i in range(0, bto.shape[0]):
            cons_1.append(wx - bto[i] * variables[i])
        cons_1.extend([-item for item in cons_1])
        for i in range(0, otw.shape[0]):
            cons_2.append(variables[i] - otw[i] * wy)
        cons_2.extend([-item for item in cons_2])
        cons = cons_1 + cons_2
        for item in cons:
            if (item > eps):
                penalty = penalty + (item - eps) * 1
        penalty = penalty + eps * eps_penalty
        return penalty


    np.random.seed(42)
    variables = np.random.uniform(low = 0.001, high = 1.0, size = bto.shape[0])
    variables = variables / np.sum(variables)
    variables = np.append(variables, [0])
    bounds    = Bounds([0]*bto.shape[0] + [0], [1]*bto.shape[0] + [1])
    w_cons    = LinearConstraint(np.append(np.ones(bto.shape[0]), [0]), [1], [1])
    results   = minimize(target_function, variables, method = 'trust-constr', bounds = bounds, constraints = [w_cons])
    weights   = results.x[:-1]
    if (verbose == True):
        print('Epsilon*:', np.round(results.x[-1], 4))
    return weights

weights = best_worst_method(bto, otw, eps_penalty = 1, verbose = True)

def fuzzy_marcos_method(dataset, criterion_type, graph = True, verbose = True):

  X = np.copy(dataset) / 1.0
  best = np.zeros(dataset.shape[1])
  worst = np.zeros(dataset.shape[1])
  best_u = np.where(criterion_type == "max", np.max(dataset_u, axis=0), np.min(dataset_u, axis=0))
  best_l = np.where(criterion_type == "max", np.max(dataset_l, axis=0), np.min(dataset_l, axis=0))
  skor_bobot = np.zeros(X.shape[1])

  for i in range(dataset.shape[1]):
    if criterion_type[i] == "max":
      best[i] = np.max(dataset[:, i])
      worst[i] = np.min(dataset[:, i])
    elif criterion_type[i] == "min":
      best[i] = np.min(dataset[:, i])
      worst[i] = np.max(dataset[:, i])

  for j in range(X.shape[1]):
    if criterion_type[j] == "max":
      X[:, j] = X[:, j] / best_u[j]
      worst[j] = worst[j] / best_u[j]
      best[j] = best[j] / best_u[j]
    elif criterion_type[j] == "min":
      X[:, j] = best_l[j] / X[:, j]
      worst[j] = best_l[j] / worst[j]
      best[j] = best_l[j] / best[j]

  return X, best, worst

n_l, best_l, worst_l = fuzzy_marcos_method(dataset_l, criterion_type)
n_m, best_m, worst_m = fuzzy_marcos_method(dataset_m, criterion_type)
n_u, best_u, worst_u = fuzzy_marcos_method(dataset_u, criterion_type)

# Menghitung mMtriks Fuzzy Berbobot V
V_l = (n_l * weights)
V_w_l = (worst_l * weights)
V_b_l = (best_l * weights)
V_m = (n_m * weights)
V_w_m = (worst_m * weights)
V_b_m = (best_m * weights)
V_u = (n_u * weights)
V_w_u = (worst_u * weights)
V_b_u = (best_u * weights)

# Jumlah Elemen Matriks Fuzzy Berbobot
S_l = np.sum(V_l, axis=1)
S_w_l = np.sum(V_w_l, axis=0)
S_b_l = np.sum(V_b_l, axis=0)
S_m = np.sum(V_m, axis=1)
S_w_m = np.sum(V_w_m, axis=0)
S_b_m = np.sum(V_b_m, axis=0)
S_u = np.sum(V_u, axis=1)
S_w_u = np.sum(V_w_u, axis=0)
S_b_u = np.sum(V_b_u, axis=0)

# Tingkat Utilitas
K_l_n = S_l / S_w_u
K_m_n = S_m / S_w_m
K_u_n = S_u / S_w_l
K_l_p = S_l / S_b_u
K_m_p = S_m / S_b_m
K_u_p = S_u / S_b_l

# Menghitung Matriks Fuzzy T
T_l = K_l_n + K_l_p
T_m = K_m_n + K_m_p
T_u = K_u_n + K_u_p

# Bilangan Fuzzy Baru
D_l = np.max(T_l)
D_m = np.max(T_m)
D_u = np.max(T_u)

# Defuzzifikasi
df_crisp = (D_l+(4*D_m)+D_u)/6

# Menentukan Fungsi Utilitas dari Solusi Anti-Ideal dan Ideal
F_K_l_n = K_l_p / df_crisp
F_K_m_n = K_m_p / df_crisp
F_K_u_n = K_u_p / df_crisp
F_K_l_p = K_l_n / df_crisp
F_K_m_p = K_m_n / df_crisp
F_K_u_p = K_u_n / df_crisp

# Defuzzifikasi Tingkat Utilitas
K_n = (K_l_n + (4 * K_m_n) + K_u_n) / 6
K_p = (K_l_p + (4 * K_m_p) + K_u_p) / 6

# Defuzzifikasi Fungsi Utilitas dari Solusi Anti-Ideal dan Ideal
F_K_n = (F_K_l_n + (4 * F_K_m_n) + F_K_u_n) / 6
F_K_p = (F_K_l_p + (4 * F_K_m_p) + F_K_u_p) / 6

# Menghitung Fungsi Utilitas
F_K = (K_p + K_n) / (1 + ((1 - F_K_p) / F_K_p) + ((1 - F_K_n) / F_K_n))

def ranking(F_K):

  # Urutkan F_K dalam urutan descending
  sorted_indices = np.argsort(F_K)[::-1]

  # Buat array peringkat
  ranks = np.zeros(F_K.shape[0])
  for i, idx in enumerate(sorted_indices):
    ranks[idx] = i + 1
  return ranks


# Memunculkan Hasil Bobot Kriteria Optimal
print("Bobot Optimal:")
for i in range(0, weights.shape[0]):
  print('w'+str(i+1)+': ', round(weights[i], 2))
ranks = ranking(F_K)

# Memunculkan Hasil Fungsi Utilitas
fungsi_utilitas = np.round(F_K, 2)
print("Fungsi Utilitas:")
print(fungsi_utilitas)

# Memunculkan Peringkat Alternatif
print("Ranking:")
for i, rank in enumerate(ranks):
  print(f"a{i+1}: {rank}")
