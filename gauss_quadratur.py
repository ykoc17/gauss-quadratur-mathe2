import numpy as np
import matplotlib as plt

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

    #Bestimme das appr. Integral
    def f_trans(x):
        return f(((a-b)/2)*x+a-((a-b)/2))*abs((a-b)/2)
    Qn = 0
    for i in range(n): Qn += wi[i]*f_trans(xi[i])

    return Qn

def gaussq_tol(f, a, b, tol):
    #Finde benötigte n mittels Fehlertoleranz-Abbruchkriteriums
    #Dann führe gaussq_n aus
    n=1
    while (abs(gaussq_n(f, a, b, n+1)-gaussq_n(f, a, b, n))>tol): n+=1
    return gaussq_n(f, a, b, n), n


###################################################################
#Definiere die Testfunktionen
def f1(x): return np.power(x, 10)
def f2(x): return np.sin(x)
def f3(x): return 1/((np.float_power(10, -2)+(np.power(x, 2))))

#Definiere Anzahl der Stützstellen n und Toleranz tol
n = 50
pow_tol = -5
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
