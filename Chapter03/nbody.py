import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


nbody_ker = ElementwiseKernel(
"float *in_x, float *in_y, float *in_v_x, float *in_v_y,  \
float *out_x, float *out_y, float *out_v_x, float *out_v_y, \
float *m, float t, int num_bodies",
"""
#define G   6.67408313e-11

float net_force_x = 0.0f;
float net_force_y = 0.0f;

for(int n=0; n < num_bodies; ++n) {
        if (n == i)
            continue;
  
        float r2 = powf(in_x[i] - in_x[n], 2.0f) + powf(in_y[i] - in_y[n], 2.0f);
        float r = sqrtf(r2);
        
        float force = G*m[i]*m[n] / r2;
        
        float force_x = force * ( in_x[n] - in_x[i]  ) / r;
        float force_y = force * ( in_y[n] - in_y[i]  ) / r;
        
        net_force_x += force_x;
        net_force_y += force_y;
        
}

float a_x = net_force_x / m[i];
float a_y = net_force_y / m[i];

out_x[i] = in_x[i] + in_v_x[i]*t + 0.5f * a_x * powf(t,2.0f);
out_y[i] = in_y[i] + in_v_y[i]*t + 0.5f * a_y * powf(t,2.0f);

out_v_x[i] = in_v_x[i] + a_x*t;
out_v_y[i] = in_v_y[i] + a_y*t;
""",
"nbody_ker")

REZ = 128
NUM_BODIES=np.int32(4000)
t=np.float32(0.005)

in_x = gpuarray.to_gpu(np.float32(np.random.random(NUM_BODIES) + .5 ))
in_y = gpuarray.to_gpu(np.float32(np.random.random(NUM_BODIES) + .5))
in_v_x = gpuarray.to_gpu(np.float32(np.random.random(NUM_BODIES) - .5))
in_v_y = gpuarray.to_gpu(np.float32(np.random.random(NUM_BODIES) - .5))

out_x = gpuarray.empty_like(in_x)
out_y = gpuarray.empty_like(in_y)
out_v_x = gpuarray.empty_like(in_v_x)
out_v_y = gpuarray.empty_like(in_v_y)

masses = np.float32(np.random.random(NUM_BODIES)*10000)
m = gpuarray.to_gpu(masses)


def xy_to_img(x_coords, y_coords, masses):
    
    img_out = np.zeros((2*REZ,2*REZ), dtype=np.int32)
    
    for x, y, mass in zip(x_coords, y_coords, masses):
        if (x < 0 or y < 0 or not np.isfinite(x) or not np.isfinite(y)):
            continue
        int_x = int(np.round(x * REZ))
        int_y = int(np.round(y * REZ))
        
        if (int_x < 2*REZ and int_y < 2*REZ):
            img_out[int_x, int_y] += int(mass)
            
    return img_out

def update_gpu(frameNum, img, in_x, in_y, in_v_x, in_v_y, out_x, out_y, out_v_x, out_v_y,t, NUM_BODIES, masses):
    
    if frameNum % 2 == 0:
        nbody_ker(in_x,in_y,in_v_x,in_v_y,out_x,out_y,out_v_x,out_v_y,m,t,NUM_BODIES)
        img.set_data(xy_to_img(out_x.get(), out_y.get(), masses))
    else:
        nbody_ker(out_x,out_y,out_v_x,out_v_y,in_x,in_y,in_v_x,in_v_y,m,t,NUM_BODIES)
        img.set_data(xy_to_img(in_x.get(), in_y.get(), masses))

    return img
    

if __name__ == '__main__':   

    fig, ax = plt.subplots()
    img = ax.imshow( xy_to_img(in_x.get(), in_y.get(), masses)  , interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img, in_x, in_y, in_v_x, in_v_y, out_x, out_y, out_v_x, out_v_y, t, NUM_BODIES, masses) , interval=0, frames=100, save_count=100)    
     
    plt.show()
