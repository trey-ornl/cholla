#include <cassert>

#include "LF2_1D_cuda.h"
#include "../utils/gpu.hpp"

static int constexpr block = 256;

__global__ static void step0(int const nx, Real const gm1, Real const *const q, Real const *const r, Real const *const s, Real *const __restrict__ q0, Real *const __restrict__ r0, Real *const __restrict__ s0, Real *const __restrict__ fq0, Real *const __restrict__ fr0, Real *const __restrict__ fs0)
{
  int const i = blockIdx.x*block+threadIdx.x;
  if (i >= nx) return;

  q0[i] = q[i];
  r0[i] = r[i];
  s0[i] = s[i];
  Real const u0 = s[i]/r[i];
  Real const p0 = gm1*(q[i]-0.5*s[i]*u0);
  fq0[i] = u0*(q[i]+p0);
  fr0[i] = s0[i];
  fs0[i] = u0*s[i]+p0;
}

__global__ static void step1(int const nx, Real const gm1, Real const dt_2dx, Real const *const q0, Real const *const r0, Real const *const s0, Real const *const fq0, Real const *const fr0, Real const *const fs0, Real *const __restrict__ q1, Real *const __restrict__ r1, Real *const __restrict__ s1, Real *const __restrict__ fq1, Real *const __restrict__ fr1, Real *const __restrict__ fs1)
{
  int const i = blockIdx.x*block+threadIdx.x;
  if (i >= nx) return;

  int const im = (i > 0 ? i-1 : 0);
  int const ip = (i < nx-1 ? i+1 : nx-1);

  q1[i] = 0.5*(q0[ip]+q0[im])-dt_2dx*(fq0[ip]-fq0[im]);
  r1[i] = 0.5*(r0[ip]+r0[im])-dt_2dx*(fr0[ip]-fr0[im]);
  s1[i] = 0.5*(s0[ip]+s0[im])-dt_2dx*(fs0[ip]-fs0[im]);

  Real const u1 = s1[i]/r1[i];
  Real const p1 = gm1*(q1[i]-0.5*s1[i]*u1);
  fq1[i] = u1*(q1[i]+p1);
  fr1[i] = s1[i];
  fs1[i] = u1*s1[i]+p1;
}

__global__ static void step2(int const nx, Real const dt_4dx, Real const *const q0, Real const *const r0, Real const *const s0, Real const *const fq0, Real const *const fr0, Real const *const fs0, Real const *const q1, Real const *const r1, Real const *const s1, Real const *const fq1, Real const *const fr1, Real const *const fs1, Real *const __restrict__ q, Real *const __restrict__ r, Real *const __restrict__ s)
{
  int const i = blockIdx.x*block+threadIdx.x;
  if (i >= nx) return;

  int const im = (i > 0 ? i-1 : 0);
  int const ip = (i < nx-1 ? i+1 : nx-1);

  q[i] = q0[i]+0.25*(q0[im]-q1[im]+q0[ip]-q1[ip]-2.0*(q0[i]-q1[i]))-dt_4dx*(fq0[ip]+fq1[ip]-fq0[im]-fq1[im]);
  r[i] = r0[i]+0.25*(r0[im]-r1[im]+r0[ip]-r1[ip]-2.0*(r0[i]-r1[i]))-dt_4dx*(fr0[ip]+fr1[ip]-fr0[im]-fr1[im]);
  s[i] = s0[i]+0.25*(s0[im]-s1[im]+s0[ip]-s1[ip]-2.0*(s0[i]-s1[i]))-dt_4dx*(fs0[ip]+fs1[ip]-fs0[im]-fs1[im]);
}

static Real *q0 = nullptr;
static Real *r0 = nullptr;
static Real *s0 = nullptr;

static Real *fq0 = nullptr;
static Real *fr0 = nullptr;
static Real *fs0 = nullptr;

static Real *q1 = nullptr;
static Real *r1 = nullptr;
static Real *s1 = nullptr;

static Real *fq1 = nullptr;
static Real *fr1 = nullptr;
static Real *fs1 = nullptr;

void LF2_Algorithm_1D_CUDA(Real *const energy, Real *const density, Real *const momentum, int const nx, Real const dx, Real const dt)
{
  if (q0 == nullptr) {
    void *ptr = nullptr;
    size_t const bytes = (12*nx)*sizeof(Real);
    GPU_Error_Check(cudaMalloc(&ptr,bytes));
    q0 = reinterpret_cast<Real *>(ptr);
    r0 = q0+nx;
    s0 = r0+nx;
    fq0 = s0+nx;
    fr0 = fq0+nx;
    fs0 = fr0+nx;
    q1 = fs0+nx;
    r1 = q1+nx;
    s1 = r1+nx;
    fq1 = s1+nx;
    fr1 = fq1+nx;
    fs1 = fr1+nx;
  }
  int const grid = (nx-1)/block+1;
  Real const dt_2dx = dt/(2.0*dx); 
  Real const dt_4dx = 0.5*dt_2dx;
  Real const gm1 = gama-1.0;
  step0<<<grid,block>>>(nx,gm1,energy,density,momentum,q0,r0,s0,fq0,fr0,fs0);
  step1<<<grid,block>>>(nx,gm1,dt_2dx,q0,r0,s0,fq0,fr0,fs0,q1,r1,s1,fq1,fr1,fs1);
  step2<<<grid,block>>>(nx,dt_4dx,q0,r0,s0,fq0,fr0,fs0,q1,r1,s1,fq1,fr1,fs1,energy,density,momentum);
}

void Free_Memory_LF2_1D()
{
  if (q0 != nullptr) {
    GPU_Error_Check(cudaFree(q0));
    q0 = nullptr;
    r0 = nullptr;
    s0 = nullptr;
    fq0 = nullptr;
    fr0 = nullptr;
    fs0 = nullptr;
    q1 = nullptr;
    r1 = nullptr;
    s1 = nullptr;
    fq1 = nullptr;
    fr1 = nullptr;
    fs1 = nullptr;
  }
}



