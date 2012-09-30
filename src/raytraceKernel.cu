// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include <cutil_math.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

__device__ int numRaysDevice = 0;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, float x, float y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;
  //r.direction = glm::vec3(0,0,-1);

  glm::vec3 A = glm::normalize(glm::cross(view, up));
  glm::vec3 B = glm::normalize(glm::cross(A, view));

  float tanVert = tan(fov.y*PI/180);
  float tanHor = tan(fov.x*PI/180);

  float camDistFromScreen = (float)((resolution.y/2.0)/tanVert);
  glm::vec3 C = view*camDistFromScreen;
  glm::vec3 M = eye + C;

  //glm::vec3 H = A * (camDistFromScreen * tanHor);
  //glm::vec3 V = B * (camDistFromScreen * tanVert);

  glm::vec3 point = M + A*(resolution.x/2.0f)*(2.0f*(x/(float)resolution.x) - 1) + B*(resolution.y/2.0f)*(2.0f*(y/(float)resolution.y) - 1);
  r.direction = glm::normalize(point - eye);

  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int oldIndex = x + (y * resolution.x);
  int newIndex = resolution.x-x + ((resolution.y - y) * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[oldIndex].x*255.0;
      color.y = image[oldIndex].y*255.0;
      color.z = image[oldIndex].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[newIndex].w = 0;
      PBOpos[newIndex].x = color.x;     
      PBOpos[newIndex].y = color.y;
      PBOpos[newIndex].z = color.z;
  }
}

__global__ void ShowTotalRays()
{
	printf("Number: %d", numRaysDevice);
}

__global__ void ResetTotalRaysCount()
{
	numRaysDevice = 0;
}
  
/*
__device__ int CalculateTotalRays(int rayCnt)
{
	  numRays = atomicAdd(&numRays, rayCnt);
}*/

__device__ float GetIntersectionValues(ray rayToBeTraced, material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms,
	int& minGeomNum, int& materialId, glm::vec3& minIntersectionPoint, glm::vec3& minNormal)
{
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	float t = -1;

	for (int geomNum = 0; geomNum < numberOfGeoms; ++geomNum) {
		float tTemp = findIntersection(geoms[geomNum], rayToBeTraced, intersectionPoint, normal);
		if (tTemp > 0.001f) {
			if (t < 0 || tTemp < t) {
				t = tTemp;
				materialId = geoms[geomNum].materialid;
				minIntersectionPoint = intersectionPoint;
				minNormal = normal;
				minGeomNum = geomNum;
			}
		}
	}

	return t;
}

__device__ glm::vec3 LightFeeler(glm::vec3 minIntersectionPoint, glm::vec3 minNormal, int minGeom,
							staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, material currMaterial, glm::vec3 rayDirection, 
							int randNum, float reductionFactor)
{
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	glm::vec3 color = glm::vec3(0,0,0);

	float Ka = 0.3; // ambient light factor
	float Kd = 0.8f;
	float Ks = 0.8f;		

	color = Ka * currMaterial.color;

	// Shadow Feeler
	int lightNum = -1;
	for (int lightNumber=0; lightNumber < numberOfGeoms; ++lightNumber)
	{
		if (materials[geoms[lightNumber].materialid].emittance > 0)
		{
			lightNum = lightNumber;
		}
	}
		if (lightNum >= 0)
		{
			/* Create pseudo - random number generator */
			float seed = randNum * 1.0f;
			int totalLightRays = 1;
			for (int numLightRays = 1; numLightRays <= totalLightRays; ++numLightRays)
			{
				seed += 1.5f;
				//glm::vec3 lightPos = getRandomPointOnObject(geoms[lightNum], seed);
				glm::vec3 lightPos = glm::vec3(0,9.2,0.0f);
				glm::vec3 lightColor = glm::vec3(1,1,1);
				glm::vec3 lightDir = minIntersectionPoint - lightPos;
				ray lightRay;
				lightRay.origin = minIntersectionPoint;
				lightRay.direction = glm::normalize(lightDir*(-1.0f));
				float tLightIntersect = -1;
				bool blocked = false;
				glm::vec3 lightColorTemp = lightColor;
				for (int geomNum = 0; geomNum < numberOfGeoms; ++geomNum)
				{
					if (materials[geoms[geomNum].materialid].emittance > 0 && geomNum != minGeom)
					{
						continue;
					}
					else
					{
						tLightIntersect = findIntersection(geoms[geomNum], lightRay, intersectionPoint, normal);
						if(tLightIntersect > 0)
						{
							if ((glm::length(lightDir) + 0.001) > tLightIntersect)
							{
								// Blocked by other object
								if (materials[geoms[geomNum].materialid].hasRefractive > 0)
								{
									// Object is transparent
									lightColorTemp *= (materials[geoms[geomNum].materialid].color*0.9f);
								}
								else
								{
									blocked = true;
									break;
								}
							}
						}
					}
				}

				lightColor = lightColorTemp;

				if (!blocked)
				{
					float diffuseComponent = Kd*glm::dot(glm::normalize(lightRay.direction), minNormal);
					float specDotProd = glm::dot(calculateReflectionDirection(minNormal, lightRay.direction), rayDirection);//rayToBeTraced.rayValue.direction);
					if (specDotProd < 0.0f)
						specDotProd = 0.0f;

					float specularComponent = 0;
					if (currMaterial.specularExponent > 0)
						specularComponent = Ks*pow(specDotProd, currMaterial.specularExponent);
			
					glm::vec3 newColor = lightColor* (reductionFactor)*(currMaterial.color*diffuseComponent + currMaterial.specularColor*specularComponent)/(float)totalLightRays;
					color += newColor;
				}
			}
		}

	return color;
}


__device__ glm::vec3 FindNewRaysAndCalculateColor(RayInPackage rayToBeTraced, RayOutPackage& rayOutPackage, material* materials, int numberOfMaterials,
	staticGeom* geoms, int numberOfGeoms, int minGeomNum, int materialId, glm::vec3 minIntersectionPoint, glm::vec3 minNormal, int randNum)
{
	glm::vec3 intersectionPoint;
	glm::vec3 normal;

	rayOutPackage.numRays = 0;
	rayOutPackage.color = glm::vec3(0,0,0);
	rayOutPackage.reductionFactor = rayToBeTraced.reductionFactor;

	// Get the refractive index. If the ray is inside the object the refractive index is from object to air
	float incidentIOR, transmittedIOR;
	if (rayToBeTraced.isInside)
	{
		incidentIOR = materials[materialId].indexOfRefraction;
		transmittedIOR = 1.0;
	}
	else
	{
		incidentIOR = 1.0;
		transmittedIOR = materials[materialId].indexOfRefraction;
	}
	Fresnel fresnel;
	fresnel.reflectionCoefficient = materials[materialId].hasReflective; // use the reflection coefficient suplied if the object has no refraction
	
	if (materials[materialId].hasRefractive > 0)
	{
		fresnel = calculateFresnel(minNormal, rayToBeTraced.rayValue.direction, incidentIOR, transmittedIOR);
		// Get the refracted ray
		glm::vec3 refractedRay;
		refractedRay = calculateTransmissionDirection(minNormal, rayToBeTraced.rayValue.direction, incidentIOR, transmittedIOR);
		rayOutPackage.isTransPresent = true;
		rayOutPackage.rayValueTrans.direction = refractedRay;
		rayOutPackage.rayValueTrans.origin = minIntersectionPoint;
		rayOutPackage.index = rayToBeTraced.index;
		++rayOutPackage.numRays;
		rayOutPackage.isInsideObject = !rayToBeTraced.isInside;
		//rayOutPackage.reductionFactor *= fresnel.reflectionCoefficient;
		rayOutPackage.color = materials[materialId].color*rayOutPackage.reductionFactor;
		//atomicAdd(&numRaysDevice, 1);
	}

	if (materials[materialId].hasReflective > 0)
	{
		// Get the reflected ray
		glm::vec3 reflectedRay = calculateReflectionDirection(minNormal, rayToBeTraced.rayValue.direction);
		rayOutPackage.isPresent = true;
		rayOutPackage.rayValue.direction = reflectedRay;
		rayOutPackage.rayValue.origin = minIntersectionPoint;
		rayOutPackage.index = rayToBeTraced.index;
		++rayOutPackage.numRays;
		rayOutPackage.reductionFactor *= fresnel.reflectionCoefficient;
		rayOutPackage.color = materials[materialId].color*rayOutPackage.reductionFactor;
		//atomicAdd(&numRaysDevice, 1);
	}

	return LightFeeler(minIntersectionPoint, minNormal, minGeomNum, geoms, numberOfGeoms, 
		materials, numberOfMaterials, materials[materialId], rayToBeTraced.rayValue.direction, randNum, rayToBeTraced.reductionFactor);
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, float offsetX, float offsetY, glm::vec3* colors, 
                            RayOutPackage* rayOutPackageList, material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	int materialId;
	int minGeomNum;
	glm::vec3 minIntersectionPoint;
	glm::vec3 minNormal;

	rayOutPackageList[index].rayValue = raycastFromCameraKernel(cam.resolution, time, ((float)x)+offsetX, ((float)y)+offsetY, cam.position, cam.view, cam.up, cam.fov);
	rayOutPackageList[index].isPresent = false;
	rayOutPackageList[index].isTransPresent = false;
	rayOutPackageList[index].index = index;
	rayOutPackageList[index].numRays = 0;
	rayOutPackageList[index].isInsideObject = false;

	float t = GetIntersectionValues(rayOutPackageList[index].rayValue, materials, numberOfMaterials, geoms, numberOfGeoms,
		minGeomNum, materialId, minIntersectionPoint, minNormal);

	RayInPackage rayInPack;
	rayInPack.index = index;
	rayInPack.isInside = false;
	rayInPack.rayValue = rayOutPackageList[index].rayValue;
	rayInPack.reductionFactor = 1.0f;
	if (t > 0 && materialId < numberOfMaterials)
	{
		colors[index] += FindNewRaysAndCalculateColor(rayInPack, rayOutPackageList[index], materials, numberOfMaterials,
			geoms, numberOfGeoms, minGeomNum, materialId, minIntersectionPoint, minNormal, index);
	}
}

__global__ void RayTrace(RayInPackage* raysToBeTraced, RayOutPackage* rayOutPackageList, glm::vec3* colors, 
	material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	float t = -1;
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	int materialId;
	int minGeomNum;
	glm::vec3 minIntersectionPoint;
	glm::vec3 minNormal;

	t = GetIntersectionValues(raysToBeTraced[index].rayValue, materials, numberOfMaterials, geoms, numberOfGeoms,
		minGeomNum, materialId, minIntersectionPoint, minNormal);

	if (t > 0 && materialId < numberOfMaterials)
	{
		colors[raysToBeTraced[index].index] += FindNewRaysAndCalculateColor(raysToBeTraced[index], rayOutPackageList[index], materials, numberOfMaterials,
			geoms, numberOfGeoms, minGeomNum, materialId, minIntersectionPoint, minNormal, index);
	}
}

__global__ void ResetColors(glm::vec3* color, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	color[index] = glm::vec3(0,0,0);
}

__global__ void CalculateColorsForAntiAliasing(glm::vec3* finalColor, glm::vec3* color, float weight, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	finalColor[index] += color[index]*weight;
}

__global__ void ClampColors(glm::vec3* color, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	
	if (color[index].x > 1)
		color[index].x = 1;
	if (color[index].y > 1)
		color[index].y = 1;
	if (color[index].z > 1)
		color[index].z = 1;

	if (color[index].x < 0)
		color[index].x = 0;
	if (color[index].y < 0)
		color[index].y = 0;
	if (color[index].z < 0)
		color[index].z = 0;
}


//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials,
	geom* geoms, int numberOfGeoms, PointLight* pointLights, int numberOfPointLights){

	//int deviceID = -1; 
	//   if(cudaSuccess == cudaGetDevice(&deviceID)) 
	//   { 
	//       cudaDeviceProp devprop; 
	//       cudaGetDeviceProperties(&devprop, deviceID); 
	//	std::cout << "Thread Per Block: " << devprop.maxThreadsPerBlock << std::endl;
	//}

	float offsetX = 0;
	float offsetY = 0;
	glm::vec3* cudaFinalImage = NULL;
	cudaMalloc((void**)&cudaFinalImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	int antiAliasIter = 1;

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	ResetColors<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaFinalImage, renderCam->resolution);

	bool antiAlias = false;
	while (antiAliasIter <= 5)
	{
		float weight;
		if (antiAlias)
		{
			if (antiAliasIter == 1)
			{
				offsetX = 0;
				offsetY = 0;
				weight = 1.0f/2.0f;
			}
			else
			{
				weight = 1.0f/8.0f;
			}

			if (antiAliasIter == 2)
			{
				offsetX = -0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
				offsetY = -0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
			}
			else if (antiAliasIter == 3)
			{
				offsetX = -0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
				offsetY = 0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
			}
			else if (antiAliasIter == 4)
			{
				offsetX = 0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
				offsetY = -0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
			}
			else if (antiAliasIter == 5)
			{
				offsetX = 0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
				offsetY = 0.33f;//(((float)(rand()%100)+0.01f)/200.0f);
			}

			++antiAliasIter;
		}
		else
		{
			antiAliasIter = 6;
			weight = 1.0;
		}

		int traceDepth = 1; //determines how many bounces the raytracer traces

		//send image to GPU
		glm::vec3* cudaimage = NULL;
		cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
		//cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
		ResetColors<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaimage, renderCam->resolution);

		// Create a rayPackage
		// The rayPackage struct will store the ray details which can be then called using the kernel to be parallely processed
		int totalPixels = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
		RayOutPackage* rayPackReset = new RayOutPackage[totalPixels];
		for (int cnt=0; cnt<totalPixels; ++cnt)
		{
			rayPackReset[cnt].isPresent = false;
			rayPackReset[cnt].index = -1;
			rayPackReset[cnt].rayValue.origin = glm::vec3(0,0,0);
			rayPackReset[cnt].rayValue.direction = glm::vec3(0,0,0);
			rayPackReset[cnt].isTransPresent = false;
			rayPackReset[cnt].isInsideObject = false;
			rayPackReset[cnt].rayValueTrans.origin = glm::vec3(0,0,0);
			rayPackReset[cnt].rayValueTrans.direction = glm::vec3(0,0,0);
			rayPackReset[cnt].color = glm::vec3(0,0,0);
		}

		//package geometry and materials and sent to GPU
		staticGeom* geomList = new staticGeom[numberOfGeoms];
		for(int i=0; i<numberOfGeoms; i++){
			staticGeom newStaticGeom;
			newStaticGeom.type = geoms[i].type;
			newStaticGeom.materialid = geoms[i].materialid;
			newStaticGeom.translation = geoms[i].translations[frame];
			newStaticGeom.rotation = geoms[i].rotations[frame];
			newStaticGeom.scale = geoms[i].scales[frame];
			newStaticGeom.transform = geoms[i].transforms[frame];
			newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
			newStaticGeom.inverseTransposeTransform = geoms[i].inverseTransposeTransforms[frame];
			geomList[i] = newStaticGeom;
		}

		staticGeom* cudageoms = NULL;
		cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
		cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

		material* cudaMaterials = NULL;
		cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
		cudaMemcpy(cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

		PointLight* cudaPointLights = NULL;
		cudaMalloc((void**)&cudaPointLights, numberOfPointLights*sizeof(PointLight));
		cudaMemcpy(cudaPointLights, pointLights, numberOfPointLights*sizeof(PointLight), cudaMemcpyHostToDevice);

		//package camera
		cameraData cam;
		cam.resolution = renderCam->resolution;
		cam.position = renderCam->positions[frame];
		cam.view = renderCam->views[frame];
		cam.up = renderCam->ups[frame];
		cam.fov = renderCam->fov;


		RayOutPackage* rayOutPackageList = NULL;
		cudaMalloc((void**)&rayOutPackageList, totalPixels*sizeof(RayOutPackage));
		cudaMemcpy(rayOutPackageList, rayPackReset, totalPixels*sizeof(RayOutPackage), cudaMemcpyHostToDevice);

		//kernel launches
		raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, offsetX, offsetY, 
									cudaimage, rayOutPackageList, cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms);
		++traceDepth;

		// Loop through rays
		int maxRayDepth = 10;
		int totalRays = totalPixels;
		while (traceDepth <= maxRayDepth && totalRays > 0)
		{
			RayOutPackage* rayPack = new RayOutPackage[totalRays];
			cudaMemcpy( rayPack, rayOutPackageList, totalRays*sizeof(RayOutPackage), cudaMemcpyDeviceToHost);

			RayInPackage* rayTracedHost = new RayInPackage[totalRays];
			int rayIndex = 0;
			for (int cnt=0; cnt<totalRays; ++cnt)
			{
				if (rayPack[cnt].numRays > 0)
				{
					if (rayPack[cnt].isPresent)
					{
						rayTracedHost[rayIndex].rayValue = rayPack[cnt].rayValue;
						rayTracedHost[rayIndex].index = rayPack[cnt].index;
						rayTracedHost[rayIndex].isInside = false;
						rayTracedHost[rayIndex].reductionFactor = rayPack[cnt].reductionFactor;
						++rayIndex;
					}
					if (rayPack[cnt].isTransPresent)
					{
						rayTracedHost[rayIndex].rayValue = rayPack[cnt].rayValueTrans;
						rayTracedHost[rayIndex].index = rayPack[cnt].index;
						rayTracedHost[rayIndex].isInside = rayPack[cnt].isInsideObject;
						rayTracedHost[rayIndex].reductionFactor = rayPack[cnt].reductionFactor;
						++rayIndex;
					}
				}
			}

			if (rayIndex <= 0)
			{
				// Break out of here as there are no more rays to be processed
				delete[] rayTracedHost;
				delete[] rayPack;
				break;
			}

			RayInPackage* raysToBeTraced = NULL;
			cudaMalloc((void**)&raysToBeTraced, rayIndex*sizeof(RayInPackage));
			cudaMemcpy(raysToBeTraced, rayTracedHost, rayIndex*sizeof(RayInPackage), cudaMemcpyHostToDevice);

			cudaFree(rayOutPackageList);
			cudaMalloc((void**)&rayOutPackageList, rayIndex*sizeof(RayOutPackage));
			cudaMemcpy(rayOutPackageList, rayPackReset, rayIndex*sizeof(RayOutPackage), cudaMemcpyHostToDevice);

			tileSize = 16;
			int threadsPerBlockForRays = tileSize;
			int fullBlocksPerGridForRays = ceil(float(rayIndex)/float(tileSize));
			ResetTotalRaysCount<<<1,1>>>();
			RayTrace<<<fullBlocksPerGridForRays, threadsPerBlockForRays>>>(raysToBeTraced, rayOutPackageList, 
										cudaimage, cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms);
			++traceDepth;
			totalRays = rayIndex;

			cudaFree(raysToBeTraced);
			delete[] rayPack;
			delete[] rayTracedHost;
		}

		CalculateColorsForAntiAliasing<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaFinalImage, cudaimage, weight, renderCam->resolution);
		
		cudaFree( rayOutPackageList);
		cudaFree( cudaimage );
		cudaFree( cudageoms );
		cudaFree( cudaMaterials );
		cudaFree( cudaPointLights );
		delete[] geomList;
		delete[] rayPackReset;

		// make certain the kernel has completed 
		cudaThreadSynchronize();
	}

	ClampColors<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaFinalImage, renderCam->resolution);

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaFinalImage);

	//retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaFinalImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	//int a;
	//std::cin >> a;
	//free up stuff, or else we'll leak memory like a madman
	
	cudaFree( cudaFinalImage );

	// make certain the kernel has completed 
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
