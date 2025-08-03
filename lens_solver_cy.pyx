# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.math cimport fabs
from libc.math cimport M_PI

from .sl_profiles import deVaucouleurs as deV

cdef double PI = M_PI

cpdef double alpha_star_unit(double x, double Re, double s_cr):
    cdef double m2d_star = deV.fast_M2d(fabs(x)/Re)
    if x == 0:
        return 0.0
    return m2d_star / (PI * x * s_cr)

cpdef double alpha_halo(double x, double[:] Rkpc, double[:] Sigma, double s_cr):
    cdef double absx = fabs(x)
    cdef Py_ssize_t n = Rkpc.shape[0]
    cdef double m2d = 0.0
    cdef double r0, r1, s0, s1, slope, sabs
    cdef Py_ssize_t i
    for i in range(n-1):
        r0 = Rkpc[i]
        r1 = Rkpc[i+1]
        if r1 <= absx:
            s0 = Sigma[i]*r0
            s1 = Sigma[i+1]*r1
            m2d += 0.5*(s0+s1)*(r1-r0)
        else:
            if r0 < absx <= r1:
                s0 = Sigma[i]*r0
                s1 = Sigma[i+1]*r1
                slope = (s1 - s0)/(r1 - r0)
                sabs = s0 + slope*(absx - r0)
                m2d += 0.5*(s0 + sabs)*(absx - r0)
            break
    if x == 0:
        return 0.0
    return (2.0*PI*m2d) / (PI * x * s_cr)

cdef double alpha_total(double x, double[:] Rkpc, double[:] Sigma,
                        double M_star, double Re, double s_cr):
    return alpha_halo(x, Rkpc, Sigma, s_cr) + M_star * alpha_star_unit(x, Re, s_cr)

cpdef tuple solve_single_lens(double[:] Rkpc, double[:] Sigma,
                              double M_star, double Re, double s_cr,
                              double beta, double einstein_radius,
                              double caustic_max_at_lens_plane,
                              int max_iter=100, double tol=1e-6):
    cdef double xA_low = einstein_radius
    cdef double xA_high = einstein_radius * 100.0
    cdef double xA_mid, f_low, f_mid
    cdef int i
    for i in range(max_iter):
        xA_mid = 0.5*(xA_low + xA_high)
        f_low = alpha_total(xA_low, Rkpc, Sigma, M_star, Re, s_cr) - xA_low + beta
        f_mid = alpha_total(xA_mid, Rkpc, Sigma, M_star, Re, s_cr) - xA_mid + beta
        if f_low * f_mid > 0:
            xA_low = xA_mid
        else:
            xA_high = xA_mid
        if fabs(xA_high - xA_low) < tol:
            break
    cdef double xA = 0.5*(xA_low + xA_high)

    cdef double xB_low = -einstein_radius
    cdef double xB_high = -caustic_max_at_lens_plane
    cdef double xB_mid
    for i in range(max_iter):
        xB_mid = 0.5*(xB_low + xB_high)
        f_low = alpha_total(xB_low, Rkpc, Sigma, M_star, Re, s_cr) - xB_low + beta
        f_mid = alpha_total(xB_mid, Rkpc, Sigma, M_star, Re, s_cr) - xB_mid + beta
        if f_low * f_mid > 0:
            xB_low = xB_mid
        else:
            xB_high = xB_mid
        if fabs(xB_high - xB_low) < tol:
            break
    cdef double xB = 0.5*(xB_low + xB_high)
    return xA, xB
