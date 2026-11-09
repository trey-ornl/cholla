#include <cassert>

#include "LF_1D_cuda.h"
#include "../utils/gpu.hpp"

static int constexpr block = 256;

__global__ static void copy(int const nx, Real const *const qi, Real const *const ri, Real const *const si, Real *const __restrict__ qo, Real *const __restrict__ ro, Real *const __restrict__ so)
{
  int const i = blockIdx.x*block+threadIdx.x;
  if (i >= nx) return;
  qo[i] = qi[i];
  ro[i] = ri[i];
  so[i] = si[i];
}

__global__ static void step(int const nx, Real const gm1, Real const dt_2dx, Real const *const qi, Real const *const ri, Real const *const si, Real *const __restrict__ qo, Real *const __restrict__ ro, Real *const __restrict__ so)
{
  int const i = blockIdx.x*block+threadIdx.x;
  if (i >= nx) return;

  int const im = (i > 0 ? i-1 : 0);
  int const ip = (i < nx-1 ? i+1 : nx-1);

  Real const qm = qi[im];
  Real const rm = ri[im];
  Real const sm = si[im];

  Real const o_rm = 1.0/rm;
  Real const um = sm*o_rm;
  Real const pm = gm1*(qm-0.5*sm*um);
  Real const fqm = um*(qm+pm);
  Real const frm = sm;
  Real const fsm = um*sm+pm;

  Real const qp = qi[ip];
  Real const rp = ri[ip];
  Real const sp = si[ip];

  Real const o_rp = 1.0/rp;
  Real const up = sp*o_rp;
  Real const pp = gm1*(qp-0.5*sp*up);
  Real const fqp = up*(qp+pp);
  Real const frp = sp;
  Real const fsp = up*sp+pp;

  qo[i] = 0.5*(qp+qm)-dt_2dx*(fqp-fqm);
  ro[i] = 0.5*(rp+rm)-dt_2dx*(frp-frm);
  so[i] = 0.5*(sp+sm)-dt_2dx*(fsp-fsm);
}

static Real *qa = nullptr;
static Real *ra = nullptr;
static Real *sa = nullptr;

void LF_Algorithm_1D_CUDA(Real *const energy, Real *const density, Real *const momentum, int const nx, Real const dx, Real const dt)
{
  if (qa == nullptr) {
    void *ptr = nullptr;
    size_t const bytes = (3*nx)*sizeof(Real);
    GPU_Error_Check(cudaMalloc(&ptr,bytes));
    qa = reinterpret_cast<Real *>(ptr);
    ra = qa+nx;
    sa = ra+nx;
  }
  int const grid = (nx-1)/block+1;
  Real const dt_2dx = dt/(2.0*dx); 
  Real const gm1 = gama-1.0;
  step<<<grid,block>>>(nx,gm1,dt_2dx,energy,density,momentum,qa,ra,sa);
  copy<<<grid,block>>>(nx,qa,ra,sa,energy,density,momentum);
}

void Free_Memory_LF_1D()
{
  if (qa != nullptr) {
    GPU_Error_Check(cudaFree(qa));
    qa = nullptr;
    ra = nullptr;
    sa = nullptr;
  }
}



