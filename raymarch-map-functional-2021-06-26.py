#!/usr/bin/env python
# coding: utf-8

#https://github.com/electricsquare/raymarching-workshop
#MIT License

#Copyright (c) 2018 Electric Square Ltd

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import math
import time
MAX_STEPS : int = 100
MAX_DIST : float = 100
SURF_DIST : float = 0.01
IMAGE_WIDTH : int = 896
IMAGE_HEIGHT : int = 414

sign = lambda x : -1 if x < 0 else (0 if x==0 else 1)
clip = lambda x,lo,hi : max(lo, min(x,hi))
fract = lambda f : f - math.floor(f)
scalar_clamp = lambda x,lo,hi : max(lo, min(x,hi))
scalar_sq = lambda x : x * x
scalar_subtract_vec3 = lambda f, tup: ( f- tup[0], f- tup[1], f- tup[2])

xz = lambda tup :(tup[0], tup[2])
xy = lambda tup :(tup[0], tup[1])
yz = lambda tup :(tup[1], tup[2])

#using pure tuples rather than list

vec2 = lambda f : (f,f)
vec2_len = lambda tup : math.sqrt( abs(tup[0]*tup[0] + tup[1]*tup[1])  )
vec2_sq_len = lambda tup : abs(tup[0]*tup[0] + tup[1]*tup[1])
vec2_add = lambda tup1, tup2 : ( tup1[0] + tup2[0], tup1[1] + tup2[1])
vec2_add_float = lambda tup1, f : ( tup1[0] + f, tup1[1] + f)
vec2_subtract = lambda tup1, tup2 : ( tup1[0] - tup2[0], tup1[1] - tup2[1])
vec2_subtract_float = lambda tup1, f : ( tup1[0] - f, tup1[1] - f)
vec2_multiply = lambda tup1, tup2 : ( tup1[0] * tup2[0], tup1[1] * tup2[1])
vec2_multiply_float = lambda tup1, f : ( tup1[0] * f, tup1[1] * f)
vec2_divide = lambda tup1, tup2 : ( tup1[0] / tup2[0], tup1[1] / tup2[1])
vec2_divide_float = lambda tup1, f : ( tup1[0] / f, tup1[1] / f)
vec2_dot = lambda tup1, tup2 : (tup1[0] * tup2[0] + tup1[1] * tup2[1])
vec2_normalize = lambda tup : vec2_multiply_float(tup, 0) if vec2_len(tup) == 0 else vec2_multiply_float(tup, 1/vec2_len(tup))
vec2_xyy = lambda tup : (tup[0], tup[1], tup[1])  
vec2_yxy = lambda tup : (tup[1], tup[0], tup[1])  
vec2_yyx = lambda tup : (tup[1], tup[1], tup[0])

vec3 = lambda f : (f,f,f)
vec3_len = lambda tup : math.sqrt( abs( tup[0]*tup[0] + tup[1]*tup[1] + tup[2]*tup[2] )  )
vec3_sq_len = lambda tup : abs(tup[0]*tup[0] + tup[1]*tup[1] + tup[2]*tup[2])
vec3_add = lambda tup1, tup2 : ( tup1[0] + tup2[0], tup1[1] + tup2[1], tup1[2] + tup2[2])
vec3_add_float = lambda tup1, f : ( tup1[0] + f, tup1[1] + f, tup1[2] + f)
vec3_subtract = lambda tup1, tup2 : ( tup1[0] - tup2[0], tup1[1] - tup2[1], tup1[2] - tup2[2])
vec3_inverse_subtract_float = lambda tup, f : ( f- tup[0], f- tup[1], f- tup[2])
vec3_subtract_float = lambda tup1, f : ( tup1[0] - f, tup1[1] - f, tup1[2] - f)
vec3_multiply = lambda tup1, tup2 : ( tup1[0] * tup2[0], tup1[1] * tup2[1], tup1[2] * tup2[2])
vec3_multiply_float = lambda tup1, f : ( tup1[0] * f, tup1[1] * f, tup1[2] * f)
vec3_divide = lambda tup1, tup2 : ( tup1[0] / tup2[0], tup1[1] / tup2[1], tup1[2] / tup2[2])
vec3_divide_float = lambda tup1, f : ( tup1[0] / f, tup1[1] / f, tup1[2] / f)
vec3_dot = lambda tup1, tup2 : (tup1[0] * tup2[0] + tup1[1] * tup2[1]+ tup1[2] * tup2[2])
vec3_norm = lambda tup : math.sqrt( vec3_dot(tup,tup) )
vec3_normalize = lambda tup : vec3_multiply_float(tup, 0) if vec3_len(tup) == 0 else vec3_multiply_float(tup, 1/vec3_len(tup))
vec3_pow = lambda tup : ( tup[0] ** tup[0], tup[1]**tup[1], tup[2]**tup[2] )
vec3_xy = lambda tup : ( tup[0], tup[1])
vec3_yz = lambda tup : ( tup[1], tup[2])
vec3_str = lambda tup : f"{tup[0]},{tup[1]},{tup[2]}"
vec3_unit_vector = lambda tup : vec3_divide_float(tup, vec2_len(tup))
vec3_cross = lambda tup1, tup2 : ( 
            (    tup1[1] * tup2[2] - tup1[2] * tup2[1])  ,
            ( - (tup1[0] * tup2[2] - tup1[2] * tup2[0]) ),
            (    tup1[0] * tup2[1] - tup1[1] * tup2[0])
        )
vec3_distance = lambda tup1,tup2: math.sqrt(abs( (tup1[0]-tup2[0])**2 + (tup1[1]-tup2[1])**2  + (tup1[2]-tup2[2])**2 )) 
# x×(1−a)+y×a
vec3_mix = lambda x , y  , a :   vec3_add ( 
    vec3_multiply(x,
        vec3_inverse_subtract_float(a,1)
    )  , 
    vec3_multiply(y,a) 
)
vec3_mix_float = lambda x , y  , a :   vec3_add ( 
    vec3_multiply_float(x,
        vec3_inverse_subtract_float(a,1)
    )  , 
    vec3_multiply_float(y,a) 
)

vec3_clamp = lambda tup, lo, hi : (
        clip(tup[0], lo, hi),
        clip(tup[1], lo, hi),
        clip(tup[2], lo, hi)
    )

smin = lambda a,b,k :   min(a,b) - scalar_sq(max(k - abs(a-b), 0)) / (k*4)  
smoothstep = lambda edge0 , edge1 , x : scalar_sq(scalar_clamp ( (x - edge0) / (edge1 - edge0), 0.0, 1.0 ))  * (3 - 2 * scalar_clamp ( (x - edge0) / (edge1 - edge0), 0.0, 1.0 ))  
scalar_mix = lambda x, y , a : x * (1-a) + y * a
scalar_step = lambda edge, x :  0 if x < edge else 1
vec3_step = lambda edge, x :  0 if x[0] < edge else 1



sdSphere = lambda center_position, radius : vec3_len(center_position) - radius
sdPlane = lambda position, plane_surface_normal, distance_from_origin : vec3_dot (position, plane_surface_normal) + distance_from_origin
#opu=operation union
operationUnion = lambda vec2a, vec2b : vec2a if vec2a[0] < vec2b[0] else vec2b
def sminCubic (a : float, b:float, k:float) -> float:
    h : float = max(k-abs(a-b), 0)
    return min(a,b) - h*h*h/ (6*k*k)

def opBlend(vec2a, vec2b):
    k = 2
    d = sminCubic(vec2a[0]. vec2b[1], k)
    #m = vec3_mix







def getCameraRayDir(uv, camPos, camTarget):
    #Calculate camera's "orthonormal basis", i.e. its transform matrix components
    camForward =vec3_normalize( vec3_subtract(camTarget, camPos) )
    camRight = vec3_normalize(vec3_cross( (0.0, 1.0, 0.0), camForward));
    camUp = vec3_normalize(vec3_cross(camForward, camRight));
     
    fPersp : float = 2.0;
    vDir = vec3_normalize(
        vec3_add(
            vec3_add(
                vec3_multiply_float(camRight, uv[0]),
                vec3_multiply_float( camUp, uv[1])
            ),
            vec3_multiply_float(camForward, fPersp )
        )
    )

 
    return vDir;

#return distance to surface 0, and material 1
#def sdf(position, iTime):
#    center_sphere = vec3_subtract(position, (3,-2.5,10))
#    return ( 
#        sdSphere ( center_sphere, 3 ),  2
#    )

def sdf(position, iTime):  
    sphere_a = sdSphere( vec3_subtract(position, (3,-2.5,10)) , 2.5 )
    sphere_b = sdSphere( vec3_subtract(position, (-3,-2.5,10)) , 2 )
    sphere_c = sdSphere( vec3_subtract(position, (0,2.5,10)) , 2.5 )
    sphere_d = sdSphere( vec3_subtract(position, (0,-0.75,10)) , 1.2 )
    plane_a = sdPlane (position, (0,1,0), 1)
    return ( min(
        sphere_a,
        sphere_b,
        sphere_c,
        sphere_d,
        plane_a
    ) , 5)
    #5 objects takes

# castRay
#     for i in step count:
#          sample scene
#              if within threshold return dist
#     return -1

#returns [0] signed distance to surface, [1] material id of hit
def castRay(rayOrigin, rayDirection, iTime) :
    max_travel_distance = 250
    distance_to_surface = 0.0; #Stores current distance along ray
    material_id = -1 # default material id

    for i in range(0,200):
        current_sdf = sdf( 
            vec3_add(
                rayOrigin,  
                vec3_multiply_float(rayDirection, distance_to_surface) 
            ) ,
            iTime
        )
        if current_sdf[0] < (0.0001*distance_to_surface):
            #when within small distance of the surface
            #count as hit
            return (distance_to_surface, material_id)
        elif current_sdf[0] > max_travel_distance:
            #did not intersect anything, speed up
            return (-1,-1)
        
        distance_to_surface += current_sdf[0]
        material_id = current_sdf[1]

    return  (distance_to_surface, material_id)



def calculateNormal(position , iTime) : 

    small_amount = 0.001 #slope
    return vec3_normalize(
        (
            sdf(vec3_add     ( position,  (small_amount,0,0) ) , iTime)[0] - 
            sdf(vec3_subtract( position,  (small_amount,0,0) ), iTime)[0], 

            sdf(vec3_add     ( position,  (0,small_amount,0) ), iTime)[0] - 
            sdf(vec3_subtract( position,  (0,small_amount,0) ), iTime)[0] , 
            
            sdf(vec3_add     ( position,  (0,0,small_amount) ), iTime)[0] - 
            sdf(vec3_subtract( position,  (0,0,small_amount) ), iTime)[0] ,            
        )
    )

# RenderRay function
#     Raymarch to find intersection of ray with scene
#     Shade
def render(rayOrigin, rayDirection, iTime):
    distance_to_surface, material_id = castRay(rayOrigin, rayDirection, iTime);
    # Visualize depth
    color = vec3(1.0-distance_to_surface*0.075)

    if material_id > -1:

        position  = vec3_add(
                    rayOrigin,  
                    vec3_multiply_float(rayDirection, distance_to_surface) 
                ) 

        if material_id > -0.5:

            color = (   0.18*material_id, 
                0.6-0.05*material_id, 
                0.2
            )

            surface_normal = calculateNormal(position, iTime)
            surface_point_to_light = vec3_normalize(
                (math.sin(iTime)*1.0, 
                math.cos(iTime*0.5)+0.5, 
                -0.5)
            )

            #surface_point_to_light = vec3_normalize( ( 1, 1, -0.8) )
            normal_dot_light = max(
                vec3_dot(surface_normal, surface_point_to_light), 0
            )
            light_direction = vec3_multiply_float(
                (0.9, 0.9, 0.8), normal_dot_light)
            
            light_ambient = (0.03, 0.04, 0.1)
            diffuse_light = vec3_multiply(color,
            vec3_add(light_ambient, light_direction) )
            return diffuse_light
        else:
            return (0.7,0.8,0)



        if t == -1: #didnt intersect 
            color = vec3_subtract_float(
                (0.30,0.36,0.60),
                rayDirection[1]*0.7
            )
        else:
            objectSurfaceColor = (0.4,0.8,0.1)
            ambient = (0.02, 0.021, 0.02)
            color = vec3_multiply(objectSurfaceColor, ambient)

            # L is vector from surface point to light, N is surface normal. N and L must be normalized!
            #float NoL = max(dot(N, L), 0.0);
            #vec3 LDirectional = vec3(0.9, 0.9, 0.8) * NoL;
            #vec3 LAmbient = vec3(0.03, 0.04, 0.1);
            #vec3 diffuse = color * (LDirectional + LAmbient);

    return color





#To calculate the each ray's direction, 
#we'll want to transform the pixel coordinate input fragCoord from the range [0, w), [0, h) 
#into [-a, a], [-1, 1], where w and h are the width and height of the screen in pixels, 
#and a is the aspect ratio of the screen. 
#We can then pass the value returned from this helper into the getCameraRayDir 
#function we defined above to get the ray direction
def normalizeScreenCoords(screenCoord, iResolution):
    return vec2_divide_float(
        vec2_subtract(
            vec2_multiply_float(screenCoord, 2),
            xy(iResolution)
        ),
        iResolution[1]
    )




# Main function
#     Evaluate camera
#     Call RenderRay
# 
def mainImage(package_iter): 
    fragCoord=package_iter[0]  
    iResolution=package_iter[1]
    iTime=package_iter[2]
    
    camPos = (0, 0, -1);
    camTarget = (0, 0, 0);
    
    uv = normalizeScreenCoords(fragCoord, iResolution);
    rayDir = getCameraRayDir(uv, camPos, camTarget);   
    
    color = render(camPos, rayDir, iTime);
    
    return color


# In[ ]:


def getPPMString(image_width:int, image_height:int, array_rgb_values):
    file_content= f"P3\n{image_width} {image_height}\n255\n"
    for i in array_rgb_values:
        file_content+=f"{i[0]} {i[1]} {i[2]}\n"
    return file_content

#The __main__ module must be importable by worker subprocesses


def main():
    nx = IMAGE_WIDTH
    ny = IMAGE_HEIGHT
    iResolution = (nx,ny,1)
    #cant use time.time since time will greatly elapse between frames?
    FRAME_PER_SECOND = 10 # 10 frames per second @ then 1 second = 10 frames
    MAX_FRAMES=6
    from timeit import default_timer as timer
    start = timer()


    array_rgb_values = []
    for frame in range(1, MAX_FRAMES ): 
        package_iter=[]
        for j in range( (ny-1), -1, -1): #j=ny-1;j>=0;j--
            for i in range( 0, nx, 1): #i=0;i<nx;i++
                fragCoord = (i,j)
                time_elapsed_seconds = time.time()
                #time_elapsed_seconds = frame / FRAME_PER_SECOND # example at frame 20, 20/10 = 2 seconds elapsed
                package_iter.append(  (fragCoord, iResolution, time_elapsed_seconds)  )
        
    end = timer()
    print(end - start, "loop gather")

    start = timer()


    for val in map(mainImage, package_iter):          
        fragColor=val
        ir = clip( int(255* fragColor[0]), 0, 255)
        ig = clip(int(255* fragColor[1]), 0, 255)
        ib= clip(int(255* fragColor[2]), 0, 255)

        array_rgb_values.append( (ir,ig,ib) )
	

    #with concurrent.futures.ProcessPoolExecutor() as executor:
        #for idx, val in enumerate(executor.map(mainImage, package_iter)):  
      #  for  val in executor.map(mainImage, package_iter):          
       #     fragColor=val
        #    ir = clip( int(255* fragColor[0]), 0, 255)
         #   ig = clip(int(255* fragColor[1]), 0, 255)
          #  ib= clip(int(255* fragColor[2]), 0, 255)

           # array_rgb_values.append( (ir,ig,ib) )
	
    end = timer()
    print(end - start, 'pop oixes') 

    start = timer()
    ppm = open(f"ray march map function 2021 06 26 takes 85 seconds.ppm", "w")
    #ppm = open(f"ray-march-multithreading.ppm", "w")
    ppm.writelines( getPPMString(IMAGE_WIDTH, IMAGE_HEIGHT, array_rgb_values) )
    ppm.close()
    del ppm
    del array_rgb_values
    end = timer()
    print(end - start, "save pixels gather") 



    
    print('done')    

main()

# In[ ]:




