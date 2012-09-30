// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient; 
};

//forward declaration
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__  bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
		float n12 = incidentIOR/transmittedIOR;

		float cos1 = glm::dot(normal, glm::vec3(-incident.x, -incident.y, -incident.z));
		float rootValue = 1 - pow(n12,2)*(1.0f-pow(cos1,2));
		if (rootValue < 0)
		{
			return calculateReflectionDirection(normal, incident);
		}

		if (cos1 > 0.0)
			return glm::normalize(normal*(n12*cos1 - sqrt(rootValue)) + incident*n12);
		else
			return glm::normalize(normal*(-n12*cos1 + sqrt(rootValue)) + incident*n12);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
	// Rr = Ri - 2N(Ri.N)
	float dotProd = glm::dot(incident, normal);
	return glm::normalize(incident - normal*2.0f*dotProd);
}

//TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
	Fresnel fresnel;

	float cosIncidence = abs(glm::dot(incident, normal));
	float sinIncidence = sqrt(1-pow(cosIncidence,2));

	if (transmittedIOR > 0.0 && incidentIOR > 0)
	{
		float commonNumerator = sqrt(1-pow(((incidentIOR/transmittedIOR)*sinIncidence),2));
		float RsNumerator = incidentIOR*cosIncidence-transmittedIOR*commonNumerator;
		float RsDenominator = incidentIOR*cosIncidence+transmittedIOR*commonNumerator;
		float Rs = pow((RsNumerator/RsDenominator),2);

		float RpNumerator = (incidentIOR * commonNumerator) - (transmittedIOR * cosIncidence);
		float RpDenominator = (incidentIOR * commonNumerator) + (transmittedIOR * cosIncidence);
		float Rp = pow((RpNumerator/RpDenominator),2);
		
		fresnel.reflectionCoefficient = (Rs + Rp)/2.0;
		fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
	}
	else
	{
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
	}
	return fresnel;
}

//LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    //crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    //Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    //Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1)); 
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}

//TODO: IMPLEMENT THIS FUNCTION
//Now that you know how cosine weighted direction generation works, try implementing non-cosine (uniform) weighted random direction generation. 
//This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
	float z = 1.0f - 2.0f*xi1;
	float temp = 1.0f - z*z;
    float r;
	if (temp < 0.0f)
		r = 0.0f;
	else
		r = sqrtf(temp);

    float phi = 2.f * PI * xi2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return glm::normalize(glm::vec3(x, y, z));
}

//TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
//returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor, 
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, 
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){

  return 1;
};

#endif
    