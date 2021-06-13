import math
import time

class vec2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = math.sqrt( 
            abs(x*x + y*y)
        )


    @staticmethod
    def static_length(vec2):
        return math.sqrt( 
            abs(vec2.x*vec2.x + vec2.y*vec2.y)
        )

    def add(self,in_vec2):
        return vec2( self.x + in_vec2.x, self.y+in_vec2.y)

    def add_by_float(self,in_float):
        return vec2( self.x + in_float, self.y+ in_float, self.z+ in_float)

    def subtract(self,in_vec2):
        return vec2( self.x - in_vec2.x, self.y-in_vec2.y)
    
    def subtract_by_float(self,in_float):
        return vec2( self.x - in_float, self.y-in_float)
 
    def multiply(self,in_vec2):
        return vec2( self.x * in_vec2.x, self.y * in_vec2.y)
    
    def multiply_by_float(self, in_float):
        return vec2(self.x * in_float , self.y * in_float)

    def divide(self,in_vec2):
        return vec2( self.x / in_vec2.x, self.y / in_vec2.y)

    def divide_by_float(self, in_float):
        return vec2(self.x / in_float , self.y / in_float)

    #swizel recursion dont do in init
    def xyy(self):
        return vec3(self.x,self.y,self.y) 
    
    def yxy(self):
        return vec3(self.y,self.x,self.y) 
    
    def yyx(self):
        return vec3(self.y,self.y,self.x)        


def sdCircle(vec2_p, float_r):
    return vec2_p.length - float_r
#https://www.shaderific.com/glsl-functions
#https://threejsfundamentals.org/threejs/lessons/threejs-shadertoy.html



class vec3:
    def __init__(self, x,y,z):
        self.x=x
        self.y=y
        self.z =z
        self.r=x
        self.g=y
        self.b =z
        self.length = math.sqrt(abs(x*x +y*y + z*z)) #distance
        self.square_length = x*x +y*y + z*z

    def add(self,in_vec3):
        return vec3( self.x + in_vec3.x, self.y+in_vec3.y, self.z+in_vec3.z)

    def add_by_float(self,in_float):
        return vec3( self.x + in_float, self.y+ in_float, self.z+ in_float)

    def subtract(self,in_vec3):
        return vec3( self.x - in_vec3.x, self.y-in_vec3.y, self.z-in_vec3.z)

    def subtract_by_float(self,in_float):
        return vec3( self.x - in_float, self.y -  in_float, self.z - in_float)

    def inverse_subtract_by_float(self, in_float):
        return vec3(in_float-self.x , in_float-self.y , in_float-self.z )
 
    def multiply(self,in_vec3):
        return vec3( self.x * in_vec3.x, self.y * in_vec3.y, self.z * in_vec3.z)
    
    def multiply_by_float(self, in_float):
        return vec3(self.x * in_float , self.y * in_float, self.z * in_float)

    def divide(self,in_vec3):
        return vec3( self.x / in_vec3.x, self.y / in_vec3.y, self.z / in_vec3.z)

    def divide_by_float(self, in_float):
        return vec3(self.x / in_float , self.y / in_float, self.z / in_float)

    def pow(self,in_vec3):
        return vec3( self.x ** in_vec3.x, self.y ** in_vec3.y, self.z ** in_vec3.z)


    def xy(self):
        return vec2(self.x, self.y)

    def __str__(self):
        return f"{self.x},{self.y},{self.z}"

    @staticmethod
    def unit_vector(in_vec3): #aka normalize
        return in_vec3.divide_by_float(in_vec3.length)
    
    @staticmethod
    def dot(v1_vec3,v2_vec3) -> float:
        return v1_vec3.x * v2_vec3.x  + v1_vec3.y * v2_vec3.y + v1_vec3.z * v2_vec3.z  

    @staticmethod
    def cross(v1_vec3,v2_vec3):
        return vec3 (
            ( v1_vec3.y * v2_vec3.z - v1_vec3.z * v2_vec3.y),
            ( - (v1_vec3.x * v2_vec3.z - v1_vec3.z * v2_vec3.x) ),
            ( v1_vec3.x * v2_vec3.y - v1_vec3.y * v2_vec3.x)
        )

#https://github.com/shiva-kannan/RayTracingInOneWeekend-Python/blob/master/src/vector.py
#https://drive.google.com/drive/folders/14yayBb9XiL16lmuhbYhhvea8mKUUK77W
#https://www.shadertoy.com/view/4dSBz3
#https://vorg.github.io/pex/docs/pex-geom/Vec3.html

class ray:
    def __init__(self,ro_vec3,rd_vec3):
        self.ro=ro_vec3
        self.rd=rd_vec3

    def point_at_parameter(self,t_float):
        return self.ro.add(self.rd.multiply_by_float(t_float))
#https://docs.python.org/3/library/typing.html

class sphere:
    def __init__(self, cen,r ):
        self.center = cen
        self.radius = r

    def hit(r,t_min, t_max,_hit_record):
        oc = r.ro.subtract(center)
        a = vec3.dot(r.rd, r.rd)
        b = 2 * vec3.dot(oc, r.rd)
        c = vec3.dot(oc,oc) - radius*radius
        discriminant = b*b - 4*a*c # b^2 – 4 ac  solving quadratic
        if (discriminant < 0):
            return -1
        else:
            return (-b - math.sqrt(discriminant)) / (2 *a)



def hit_sphere(center, radius, r):
    oc = r.ro.subtract(center)
    a = vec3.dot(r.rd, r.rd)
    b = 2 * vec3.dot(oc, r.rd)
    c = vec3.dot(oc,oc) - radius*radius
    discriminant = b*b - 4*a*c # b^2 – 4 ac  solving quadratic
    if (discriminant < 0):
        return -1
    else:
        return (-b - math.sqrt(discriminant)) / (2 *a)
#https://github.com/pex-gl/pex-math
def color(r):
    t = hit_sphere( vec3(0,0,-1), 0.5, r  )
    if t > 0:
        print(t)
        n = vec3.unit_vector( r.point_at_parameter(t).subtract(vec3(0,0,-1)) )
        return n.add_by_float(1).multiply_by_float(0.5)

    #else return sky blue gradient
    unit_direction = vec3.unit_vector(r.rd)
    t = 0.5 * (unit_direction.y + 1)
    return vec3(1,1,1).multiply_by_float( (1.0-t) ).add( vec3(0.5,0.7,1.0).multiply_by_float(t) )

#https://www.hxa.name/minilight/
#ls | grep sign -r
sign = lambda x : -1 if x < 0 else (0 if x==0 else 1)
distance3 = lambda v,w : math.sqrt(abs( (v.x-w.x)**2 + (v.y-w.y)**2  + (v.z-w.z)**2 )) #distance
dot = lambda v1,v2 :  v1.x * v2.x  + v1.y * v2.y + v1.z * v2.z  

norm = lambda v : math.sqrt( dot(v,v))
print('test','2.449489743' ,distance3(vec3(1,0,5), vec3(0,2,4)) ) 

#The normalize function returns a vector with length 1.0 
# that is parallel to x, i.e. x divided by its length. 
# The input parameter can be a floating scalar or a float vector. 
# In case of a floating scalar the normalize function is trivial 
# and returns 1.0.
def normalize(v3 : vec3):
    l = math.sqrt( 
            abs(v3.x*v3.x + v3.y*v3.y + v3.z * v3.z)
        )
    if l == 0:
        return v3.multiply_by_float(0)
    else:
        return v3.multiply_by_float(1 / l)


print( normalize(vec3(3,2,-1)), 3/math.sqrt(14), math.sqrt(2/7), -(1/math.sqrt(14)))
print(normalize(vec3(5,4,2)), "0.74,0.59,0.29")

clip = lambda x,lo,hi : max(lo, min(x,hi))

clamp1 = lambda x,lo,hi : max(lo, min(x,hi))
length2 = lambda v: math.sqrt( abs(v.x*v.x + v.y*v.y) )
length3 = lambda v: math.sqrt(abs(v.x*v.x +v.y*v.y + v.z*v.z))

def clamp3(v3 : vec3 , lo, hi) -> vec3:
    #lo_v3 : vec3 = vec3(lo,lo,lo)
    #hi_v3 : vec3 = vec3(hi,hi,hi)
    return vec3 (
        clip(v3.x, lo, hi),
        clip(v3.y, lo, hi),
        clip(v3.z, lo, hi)
    )



def smoothstep(edge0 : float, edge1 : float, x : float) -> float:
    #Scale, bias and saturate x to 0..1 range
    x = clamp1((x - edge0) / (edge1 - edge0), 0.0, 1.0); 
    # Evaluate polynomial
    return x * x * (3 - 2 * x);

def mix1(x : float,y : float,a : float) -> float:
    return x * (1-a) + y * a

def mix3(x : vec3, y : vec3 , a : vec3 ) -> vec3:
    return x.multiply(a.inverse_subtract_by_float(1)).add( y.multiply(a) )

def mixv3v3f(x:vec3, y: vec3, a:float)  -> vec3:
    return x.multiply_by_float(1-a).add( y.multiply_by_float(a) )

def step1(edge : float, x : float) -> float:
	return 0 if x < edge else 1

def step3f(edge : float, x : vec3) -> vec3:
	return 0 if x.x < edge else 1





#main()
#https://www.youtube.com/watch?v=PGtv-dBi2wE

MAX_STEPS : int = 100
MAX_DIST : float = 100
SURF_DIST : float = 0.01
IMAGE_WIDTH : int = 256
IMAGE_HEIGHT : int = 256

55:47 left off

def map_scene(position :  vec3) -> float:
    distance_sphere : float= length3(position) - 0.25
    distance_plane : float = position.y - (-0.25)
    return min(distance_sphere, distance_plane)

def calcNormal(position : vec3) : 
    #derivative xyy, yxy, yyx

    small_amount = vec2(0.0001,0)
    return normalize(
        vec3(
            map_scene(position.add(small_amount.xyy())) - 
            map_scene(position.subtract(small_amount.xyy())) , 

            map_scene(position.add(small_amount.yxy())) - 
            map_scene(position.subtract(small_amount.yxy())) , 
            
            map_scene(position.add(small_amount.yyx())) - 
            map_scene(position.subtract(small_amount.yyx())) ,            
        )
    )

def castRay (ray_origin, ray_direction) -> float:
    ray_march_step : float = 0
    for i in range(100):
        position_point : vec3 = ray_origin.add(
            ray_direction.multiply_by_float(ray_march_step)
        ) #march 
        hit : float = map_scene(position_point)
        if hit < 0.001:
            break #inside
        ray_march_step += hit
        if ray_march_step > 20:
            break #too far outside
    if ray_march_step > 20:
        ray_march_step = -1
    return ray_march_step

#https://www.youtube.com/watch?v=-pdSjBPH3zM
#https://www.youtube.com/watch?v=Cfe5UQ-1L9Q
def mainImage(fragCoord : vec2, iResolution : vec3, iTime : float) -> vec3:
    
    p_pixel : vec2 = fragCoord.multiply_by_float(2).subtract(
        iResolution.xy()).divide_by_float(iResolution.y)
    camera_ro_ray_origin : vec3 = vec3(
        1 * math.sin(iTime),
        0,
        1 * math.cos(iTime)
    )  #rotate
    ta  : vec3= vec3(0,0,0)
    ww : vec3 = normalize(ta.subtract(camera_ro_ray_origin))
    uu : vec3 = normalize( vec3.cross( ww, vec3(0,1,0 ) ) ) #right vector
    vv : vec3 = normalize( vec3.cross(uu,ww) )

    rd_ray_direction : vec3 =  normalize( 
        uu.multiply_by_float(p_pixel.x).add(
            vv.multiply_by_float(p_pixel.y)
            ).add(
                ww.multiply_by_float(1.5)
            )
    )

#47.33

    #gradient base sky using y component of vector
    color_pixel : vec3= vec3(0.4, 0.75, 1).subtract_by_float(0.7 *rd_ray_direction.y )#blue sky if no hits
    color_pixel = mixv3v3f(
        color_pixel,
        vec3(0.7,0.75,0.8),
        math.exp(-10.0*rd_ray_direction.y)
    )

    ray_march_step : float = castRay(camera_ro_ray_origin, rd_ray_direction)

        
    if ray_march_step > 0: 
        #hit something
        position_point : vec3 = camera_ro_ray_origin.add(
            rd_ray_direction.multiply_by_float(ray_march_step)
        )       
        normal : vec3= calcNormal(position_point)

        material_base : vec3 = vec3(0.2,0.2,0.2) #base color

        sun_direction : vec3 = normalize(vec3(0.8,0.4,0.2))
        sun_diffuse_light : float = clamp1 ( 
            vec3.dot(normal,sun_direction), 0, 1
        )
        #position we shading to light
        sun_shadow = step1(
            castRay( 
                position_point.add(normal.multiply_by_float(0.001)), sun_direction
            ),0
        )


        sky_direction = vec3(0,1,0)
        sky_diffuse_light : float = clamp1 ( 
            0.5 + 0.5*vec3.dot(normal,sky_direction), 0, 1
        )

        #back of sphere light
        bounce_direction = vec3(0,-1,0)
        bounce_diffuse_light : float = clamp1 ( 
            0.5 + 0.5*vec3.dot(normal,bounce_direction), 0, 1
        )

        color_pixel =  material_base.multiply(
            vec3(7,4.5,3).multiply_by_float(sun_diffuse_light).multiply_by_float(sun_shadow )
        )
        color_pixel = color_pixel.add (
            material_base.multiply(vec3(0.5,0.8,0.9).multiply_by_float(sky_diffuse_light)) 
        )

        color_pixel = color_pixel.add (
            material_base.multiply(vec3(0.7,0.3,0.2).multiply_by_float(bounce_diffuse_light)) 
        )

        color_pixel = color_pixel.pow( vec3(0.4545,0.4545,0.4545)) #gamma correction brighter function
    return color_pixel

def main():
    nx = IMAGE_WIDTH
    ny = IMAGE_HEIGHT
    iResolution = vec3(nx,ny,1)
    vec3_lower_left_corner=vec3(-2,-1,-1)
    vec3_horizontal=vec3(4,0,0)
    vec3_vertical=vec3(0,2,0)
    vec3_origin=vec3 (0,0,1)

    for iTime in range(1, 60 ):
        file_content= f"P3\n{nx} {ny}\n255\n"
        for j in range( (ny-1), -1, -1): #j=ny-1;j>=0;j--
            for i in range( 0, nx, 1): #i=0;i<nx;i++
                fragCoord : vec2 = vec2(i,j)
                #u= i / nx #0 to 1 value
                #v = j / ny
                #r = ray(
                #        vec3_origin, 
                #        vec3_lower_left_corner.add( 
                #            vec3_horizontal.multiply_by_float(u)
                #        ).add(
                #            vec3_vertical.multiply_by_float(v)
                #        )
                #)
                #vec3_col = color(r)
                #iTime = time.time()
                fragColor = mainImage( fragCoord,iResolution, iTime/20)
                ir = int(255.99* fragColor.r)
                ig = int(255.99* fragColor.g)
                ib= int(255.99* fragColor.b)
                file_content+=f"{ir} {ig} {ib}\n"

        print(f"frame{iTime}")
        ppm = open(f"./animate/ray-march-sphere-march{iTime:03}.ppm", "w")
        ppm.writelines(file_content)
        ppm.close()
    
    print('done')    

main()



#if HW_PERFORMANCE==0
#define AA 1
#else
#define AA 2   // make this 2 or 3 for antialiasing
#endif

#//------------------------------------------------------------------

dot2 = lambda v  :  dot(v,v)
ndot = lambda  a , b : a.x*b.x - a.y*b.y; 

