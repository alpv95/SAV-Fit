from sympy import *
P, q, M, m_diff, Z, gc, z, g, k, x = symbols('P q M m_diff Z gc z g k x')

#Centre of mass frame
#expr1 = P - (1-k)*z - (M + m_diff) / (Z - x) + (M - m_diff) / (Z + x)
#expr1 = expr1.subs(x, q/(1-k) - (M + m_diff) / ((Z - z)*(1-k)) + (M - m_diff) / ((Z + z)*(1-k)) )
#poly1,d = fraction(cancel(expr1))
#print("CoM Frame: \n")
##print(poly1.coeff(k,3)) #0
#for i in reversed(range(7)):
#    eqn = str(poly1.coeff(z,i)).replace("M**2", "total_m_pow2")
#    eqn = eqn.replace("m_diff**2", "m_diff_pow2")
#    eqn = eqn.replace("Z**6", "z1_pow6")
#    eqn = eqn.replace("Z**5", "z1_pow5")
#    eqn = eqn.replace("Z**4", "z1_pow4")
#    eqn = eqn.replace("Z**3", "z1_pow3")
#    eqn = eqn.replace("Z**2", "z1_pow2")
#    eqn = eqn.replace("q**2", "zeta_conj_pow2")
#    eqn = eqn.replace("k**3", "K_pow3")
#    eqn = eqn.replace("k**2", "K_pow2")
#
#
#    eqn = eqn.replace("M", "total_m")
#    eqn = eqn.replace("k", "K")
#    eqn = eqn.replace("Z", "pos_z1")
#    eqn = eqn.replace("P", "zeta")
#    eqn = eqn.replace("q", "zeta_conj")
#    print("{}:".format(i), eqn)

# #Planet Frame
# expr1 = P - (1-k)*z + (M + m_diff) / (x) + (M - m_diff) / (x - Z)
# expr1 = expr1.subs(x, q/(1-k) + (M + m_diff) / ((z)*(1-k)) + (M - m_diff) / ((z - Z)*(1-k)) )
# poly1,d = fraction(cancel(xpr1))
print("Planet Frame: \n")
# for i in reversed(range(7)):
#     eqn = str(poly1.coeff(z,i)).replace("M**2", "total_m_pow2")
#     eqn = eqn.replace("m_diff**2", "m_diff_pow2")
#     #eqn = eqn.replace("Z**6", "z1_pow6")
#     #eqn = eqn.replace("Z**5", "z1_pow5")
#     #eqn = eqn.replace("Z**4", "z1_pow4")
#     eqn = eqn.replace("Z**3", "z1_pow3")
#     eqn = eqn.replace("Z**2", "z1_pow2")
#     eqn = eqn.replace("q**2", "zeta_conj_pow2")
#     #eqn = eqn.replace("k**3", "K_pow3")
#     eqn = eqn.replace("k**2", "K_pow2")


#     eqn = eqn.replace("M", "total_m")
#     eqn = eqn.replace("k", "K")
#     eqn = eqn.replace("Z", "z1")
#     eqn = eqn.replace("P", "zeta")
#     eqn = eqn.replace("q", "zeta_conj")
#     print("{}:".format(i), eqn)

# #Planet Frame with convergence single lens
#expr1 = P - (1-k)*z + 1 / (x)
#expr1 = expr1.subs(x, q/(1-k) + 1 / ((z)*(1-k)) )
#poly1,d = fraction(cancel(expr1))
#print("Planet Frame: \n")
## print("_____", poly1.coeff(q,4)) #0
## print("_____", poly1.coeff(m_diff,4)) #0
## print("_____", poly1.coeff(M,4)) #0
#for i in reversed(range(11)):
#    #print("{}".format(i), poly1.coeff(z,i).subs(k,0).subs(g,0))
#    eqn = str(poly1.coeff(z,i))
#    eqn = eqn.replace("q**3", "zeta_conj_pow3")
#    eqn = eqn.replace("q**2", "zeta_conj_pow2")
#    eqn = eqn.replace("k**3", "K_pow3")
#    eqn = eqn.replace("k**2", "K_pow2")
#
#
#    eqn = eqn.replace("k", "K")
#    eqn = eqn.replace("P", "zeta")
#    eqn = eqn.replace("q", "zeta_conj")
#    print("{}:".format(i), eqn)

# #Planet Frame with shear
# expr1 = P - (1-k)*z + g*x + (M + m_diff) / (x) + (M - m_diff) / (x - Z)
# expr1 = expr1.subs(x, q/(1-k) + gc*z/(1-k) + (M + m_diff) / ((z)*(1-k)) + (M - m_diff) / ((z - Z)*(1-k)) )
# poly1,d = fraction(cancel(expr1))
# print("Planet Frame: \n")
# # print("_____", poly1.coeff(q,4)) #0
# # print("_____", poly1.coeff(m_diff,4)) #0
# # print("_____", poly1.coeff(M,4)) #0
# for i in reversed(range(11)):
#    #print("{}".format(i), poly1.coeff(z,i).subs(k,0).subs(g,0))
#    eqn = str(poly1.coeff(z,i)).replace("M**2", "total_m_pow2")
#    eqn = eqn.replace("M**3", "total_m_pow3")
#    eqn = eqn.replace("m_diff**2", "m_diff_pow2")
#    eqn = eqn.replace("m_diff**3", "m_diff_pow3")
#    #eqn = eqn.replace("Z**6", "z1_pow6")
#    #eqn = eqn.replace("Z**5", "z1_pow5")
#    eqn = eqn.replace("Z**4", "z1_pow4")
#    eqn = eqn.replace("Z**3", "z1_pow3")
#    eqn = eqn.replace("Z**2", "z1_pow2")
#    eqn = eqn.replace("q**3", "zeta_conj_pow3")
#    eqn = eqn.replace("q**2", "zeta_conj_pow2")
#    eqn = eqn.replace("k**4", "K_pow4")
#    eqn = eqn.replace("k**3", "K_pow3")
#    eqn = eqn.replace("k**2", "K_pow2")
#    eqn = eqn.replace("g**4", "G_pow4")
#    eqn = eqn.replace("g**3", "G_pow3")
#    eqn = eqn.replace("g**2", "G_pow2")
#    eqn = eqn.replace("gc**4", "Gc_pow4")
#    eqn = eqn.replace("gc**3", "Gc_pow3")
#    eqn = eqn.replace("gc**2", "Gc_pow2")


#    eqn = eqn.replace("M", "total_m")
#    eqn = eqn.replace("k", "K")
#    eqn = eqn.replace("g", "G")
#    eqn = eqn.replace("gc", "Gc")
#    eqn = eqn.replace("Z", "z1")
#    eqn = eqn.replace("P", "zeta")
#    eqn = eqn.replace("q", "zeta_conj")
#    print("{}:".format(i), eqn)

# #Planet Frame with shear for C++
expr1 = P - (1-k)*z + g*x + (M + m_diff) / (x) + (M - m_diff) / (x - Z)
expr1 = expr1.subs(x, q/(1-k) + gc*z/(1-k) + (M + m_diff) / ((z)*(1-k)) + (M - m_diff) / ((z - Z)*(1-k)) )
poly1,d = fraction(cancel(expr1))
print("Planet Frame: \n")
# print("_____", poly1.coeff(q,4)) #0
# print("_____", poly1.coeff(m_diff,4)) #0
# print("_____", poly1.coeff(M,4)) #0
for i in reversed(range(11)):
    eqn = str(poly1.coeff(z,i)).replace("M**2", "mtot*mtot")
    eqn = eqn.replace("M**3", "mtot*mtot*mtot")
    eqn = eqn.replace("m_diff**2", "mdiff*mdiff")
    eqn = eqn.replace("m_diff**3", "mdiff*mdiff*mdiff")
    eqn = eqn.replace("Z**4", "a*a*a*a")
    eqn = eqn.replace("Z**3", "a*a*a")
    eqn = eqn.replace("Z**2", "a*a")
    eqn = eqn.replace("q**3", "yc*yc*yc")
    eqn = eqn.replace("q**2", "yc*yc")
    eqn = eqn.replace("k**4", "K*K*K*K")
    eqn = eqn.replace("k**3", "K*K*K")
    eqn = eqn.replace("k**2", "K*K")
    eqn = eqn.replace("g**4", "G*G*G*G")
    eqn = eqn.replace("g**3", "G*G*G")
    eqn = eqn.replace("g**2", "G*G")
    eqn = eqn.replace("gc**4", "Gc*Gc*Gc*Gc")
    eqn = eqn.replace("gc**3", "Gc*Gc*Gc")
    eqn = eqn.replace("gc**2", "Gc*Gc")


    eqn = eqn.replace("M", "mtot")
    eqn = eqn.replace("k", "K")
    eqn = eqn.replace("g", "G")
    eqn = eqn.replace("gc", "Gc")
    eqn = eqn.replace("Z", "a")
    eqn = eqn.replace("P", "y")
    eqn = eqn.replace("q", "yc")
    print("{}:".format(i), eqn)


# #Planet Frame for C++
# expr1 = P - (1-k)*z + (M + m_diff) / (x) + (M - m_diff) / (x - Z)
# expr1 = expr1.subs(x, q/(1-k) + (M + m_diff) / ((z)*(1-k)) + (M - m_diff) / ((z - Z)*(1-k)) )
# poly1,d = fraction(cancel(expr1))
# print("Planet Frame: \n")
# for i in reversed(range(7)):
#     eqn = str(poly1.coeff(z,i)).replace("M**2", "mtot*mtot")
#     eqn = eqn.replace("m_diff**2", "mdiff*mdiff")
#     eqn = eqn.replace("Z**3", "a*a*a")
#     eqn = eqn.replace("Z**2", "a*a")
#     eqn = eqn.replace("q**2", "yc*yc")
#     eqn = eqn.replace("k**2", "K*K")


#     eqn = eqn.replace("M", "mtot")
#     eqn = eqn.replace("m_diff", "mdiff")
#     eqn = eqn.replace("k", "K")
#     eqn = eqn.replace("Z", "a")
#     eqn = eqn.replace("P", "y")
#     eqn = eqn.replace("q", "yc")
#     print("{}:".format(i), eqn)
