*** Ray Tracer ***

Implementation Details
The goal was to create a ray tracer on the GPU using CUDA.

I initially thought of implrmenting parallelism using pixels directly. But then I realized then it would be a simple loop within the kernel function executing the recursion. So I tried packing up the rays and creating threads based on how many rays were currently available. This greatly decreased the number of threads being used as most of the diffuse surfaces just returned color without any reflections or refraction. This also helped me implement refraction as each ray could now give rise to 2 rays.

The bad part was that I did the compaction of the list on the CPU and transferred it back. So I was gaining memory by releasing threads, but I was loosing time as I was going back to the CPU. I did not get time to implement the stream compaction on the GPU as that would have been the ideal solution. Maybe will try it in the path tracer.


Features implemented:
* Diffuse Shading
* Phong's specular shading
* Fresnel's equation to calculate transmittance and reflectance using the refractive index.
* Specular Reflection
* Refraction
* Anti-aliasing

Blog:
http://cudaraytracer.blogspot.com/

