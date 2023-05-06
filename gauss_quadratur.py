import numpy as np
import matplotlib.pyplot as plt

def gaussq_n(f, a, b, n):
    #Berechne für festes n die Stützstellen
    A = np.zeros((n,n))
    beta_i = np.zeros(n)

    for i in range(1, n): beta_i[i] = i/(np.sqrt(4*(np.power(i,2))-1))

    for piv in range(n-1): A[piv+1][piv], A[piv][piv+1] = beta_i[piv+1], beta_i[piv+1]

    xi, Z = np.linalg.eig(A)

    #Berechne die entspechenden Gewichte
    wi = np.zeros(n)
    for i in range(n): wi[i] = 2*(np.power(Z[0][i], 2))

    #Transformiere Funktion durch Substitution damit die Formeln für beliebige Grenzen gelten
    def f_trans(x):
        return f(((a-b)/2)*x+a-((a-b)/2))*abs((a-b)/2)

    #Bestimme Qn
    Qn = 0
    for i in range(n): Qn += wi[i]*f_trans(xi[i])

    return Qn

def gaussq_tol(f, a, b, tol):
    #Finde benötigte n mittels Fehlertoleranz-Abbruchkriteriums
    #Dann führe gaussq_n aus
    n=1
    while (abs(gaussq_n(f, a, b, n+1)-gaussq_n(f, a, b, n))>tol): n+=1
    return gaussq_n(f, a, b, n), n

def abs_fehler_festes_n(f, a, b, I, n_werte):
    abs_fehler_werte = np.empty(len(n_werte))
    for i in range(100): abs_fehler_werte[i] = abs(gaussq_n(f, a, b, n_werte[i]) - I)
    return abs_fehler_werte

###################################################################
#Definiere die Testfunktionen
def f1(x): return np.power(x, 10)
def f2(x): return np.sin(x)
def f3(x): return 1/((np.float_power(10, -2)+(np.power(x, 2))))

#Definiere Anzahl der Stützstellen n und Toleranz tol
n = 10
pow_tol = -2
tol = np.float_power(10, pow_tol)

###################################################################
#Aufgaben
print("a)")

Qn_n = gaussq_n(f1, -1, 1, n)
print("n =", n, ": Q_"+str(n)+"[f1]", "=", Qn_n)

Qn_tol, n_tol = gaussq_tol(f1, -1, 1, tol)
print("tol =", tol, ": Q_"+str(n_tol)+"[f1]", "=", Qn_tol)

###################################################################
print("\nb)")

Qn_n = gaussq_n(f2, 0, np.pi, n)
print("n =", n, ": Q_"+str(n)+"[f2]", "=", Qn_n)

Qn_tol, n_tol = gaussq_tol(f2, 0, np.pi, tol)
print("tol =", tol, ": Q_"+str(n_tol)+"[f2]", "=", Qn_tol)

###################################################################
print("\nc)")

Qn_n = gaussq_n(f3, -2, 3, n)
print("n =", n, ": Q_"+str(n)+"[f3]", "=", Qn_n)

Qn_tol, n_tol = gaussq_tol(f3, -2, 3, tol)
print("tol =", tol, ": Q_"+str(n_tol)+"[f3]", "=", Qn_tol)

####################################################################
#Plotten

fig, (axa, axb, axc) = plt.subplots(3,1, figsize=(8, 12))
fig.suptitle("|Qn − ∫f(x)dx| für n = 1,...,100", fontsize=16, fontweight="bold")

n_werte = np.arange(1, 101)

absf_f1 = abs_fehler_festes_n(f1, -1, 1, 2/11, n_werte)
axa.plot(n_werte, absf_f1, "o", markersize=2, color="C3")
axa.set_ylabel("abs. Fehler")

absf_f2 = abs_fehler_festes_n(f2, 0, np.pi, 2, n_werte)
axb.plot(n_werte, absf_f2, "o", markersize=2, color="C3")
axb.set_ylabel("abs. Fehler")

absf_f3 = abs_fehler_festes_n(f3, -2, 3, 10*np.arctan(20)+10*np.arctan(30), n_werte)
axc.plot(n_werte, absf_f3, "o", markersize=2, color="C3")
axc.set_xlabel("n")
axc.set_ylabel("abs. Fehler")

#####################################################################
fig2, (axat, axbt, axct) = plt.subplots(3,1, figsize=(5, 12))
fig2.suptitle("n_min für tol = 1e-1,...,1e-6", fontsize=16, fontweight="bold")

tol_werte = np.empty(6)
for i in range(len(tol_werte)): tol_werte[i] = np.float_power(10, -(i+1))

n_fuer_tol_werte1 = np.empty(6)
for i in range(len(tol_werte)): Qn_tol, n_fuer_tol_werte1[i] = gaussq_tol(f1, -1, 1, tol_werte[i])
axat.plot(tol_werte, n_fuer_tol_werte1, "x", markersize=5, color="C9")
axat.set_ylabel("n_min")
axat.set_xscale("log")
for index in range(len(tol_werte)):
  axat.text(tol_werte[index], n_fuer_tol_werte1[index], int(n_fuer_tol_werte1[index]), size=10, fontweight="bold", color="C0")

n_fuer_tol_werte2 = np.empty(6)
for i in range(len(tol_werte)): Qn_tol, n_fuer_tol_werte2[i] = gaussq_tol(f2, 0, np.pi, tol_werte[i])
axbt.plot(tol_werte, n_fuer_tol_werte2, "x", markersize=5, color="C9")
axbt.set_ylabel("n_min")
axbt.set_xscale("log")
for index in range(len(tol_werte)):
  axbt.text(tol_werte[index], n_fuer_tol_werte2[index], int(n_fuer_tol_werte2[index]), size=10, fontweight="bold", color="C0")

n_fuer_tol_werte3 = np.empty(6)
for i in range(len(tol_werte)): Qn_tol, n_fuer_tol_werte3[i] = gaussq_tol(f3, -2, 3, tol_werte[i])
axct.plot(tol_werte, n_fuer_tol_werte3, "x", markersize=5, color="C9")
axct.set_xlabel("tol")
axct.set_ylabel("n_min")
axct.set_xscale("log")
for index in range(len(tol_werte)):
  axct.text(tol_werte[index], n_fuer_tol_werte3[index], int(n_fuer_tol_werte3[index]), size=10, fontweight="bold", color="C0")

plt.show()