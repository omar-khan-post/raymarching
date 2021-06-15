import math
import time
MAX_STEPS : int = 100
MAX_DIST : float = 100
SURF_DIST : float = 0.01
IMAGE_WIDTH : int = 320
IMAGE_HEIGHT : int = 320

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

    @staticmethod
    def dot(v1,v2) -> float:
        return v1.x * v2.x  + v1.y * v2.y  
    
    @staticmethod
    def normalize( v2):
        if v2.length == 0:
            return v2.multiply_by_float(0)
        else:
            return v2.multiply_by_float(1 / v2.length) 
        #static methods doesnt have self

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
    
    def yz(self):
        return vec2(self.y, self.z)

    def setXZ(self, v:vec2):
        self.x = v.x
        self.z = v.y

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
    
    @staticmethod
    def normalize( v3):
        if v3.length == 0:
            return v3.multiply_by_float(0)
        else:
            return v3.multiply_by_float(1 / v3.length)
        


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


def fract(f : float) -> int:
    return f - math.floor(f)


#main()
def sdSphere(position : vec3, radius : float):
    return position.length - radius

def sdElipsoid(position : vec3, radius : vec3):
    k0 : float  = length3(position.divide(radius))
    k1 : float = position.divide(radius).divide(radius).length
    return k0 * (k0-1)/k1

def smin(a:float, b:float,k:float) -> float: #smooth min
    h : float= max(k - abs(a-b), 0)
    return min(a,b) - h * h / (k*4)

#55:47 left off
def sdGuy( position : vec3, iTime) -> vec2:
    fract_time = fract(iTime) #example 10 frames per second, frame would be 1,2,3, etc?
    y : float = (4.0 * fract_time * (1.0 - fract_time)) * 0.25
    dy : float   = 4 * (1-2*fract_time)
    u : vec2 = vec2.normalize(vec2(1, -dy))
    v : vec2 =  vec2(dy, 1)
    center : vec3 =  vec3(0,y,0)
    sy : float = 0.5 + 0.5 * y #squash
    sz : float = 1/sy #preserve volume when stretching
    rad : vec3 = vec3(0.25,0.25*sy,0.25*sz)
    q : vec3 = position.subtract(center)
    #q.setXZ( vec2( 
    #    vec2.dot( u, q.yz() ), 
    #    vec2.dot( v, q.yz() ) 
    #    )
    #)
    d = sdElipsoid(q, rad)
    h : vec3= q
    d2 = sdElipsoid(
        h.subtract( vec3(0,0.28,0) ), 
        vec3(0.2,0.2,0.2)
    ) #head
    d3 = sdElipsoid(
        h.subtract( vec3(0,0.28,-0.1) ), 
        vec3(0.2,0.2,0.2)
    ) #backhead
    d2 = smin(d2, d3, 0.03)
    d = smin(d, d2, 0.1)

    res : vec2 = vec2(d,2)
    #eye
    sh : vec3 = vec3(
        abs(h.x), h.y, h.z
    )
    d4 = sdSphere( 
        sh.subtract(vec3(0.08,0.28,0.16)), 0.05 
    ) 
    if d4<d:
        res = vec2(d4, 3)
    #d  = min(d, d4)
    return  res #return distance and object id

def map_scene(position :  vec3, iTime) -> vec2:
    #distance_sphere : float= length3(position) - 0.25
    distance_guy : vec2 = sdGuy(position, iTime)
    distance_plane : float = position.y - (-0.25)
    return  vec2(distance_plane,1) if distance_plane < distance_guy.x else distance_guy

def calcNormal(position : vec3, iTime) : 
    #derivative xyy, yxy, yyx

    small_amount = vec2(0.0001,0)
    return normalize(
        vec3(
            map_scene(position.add(small_amount.xyy()) , iTime).x - 
            map_scene(position.subtract(small_amount.xyy()), iTime).x , 

            map_scene(position.add(small_amount.yxy()), iTime).x - 
            map_scene(position.subtract(small_amount.yxy()), iTime).x , 
            
            map_scene(position.add(small_amount.yyx()), iTime).x - 
            map_scene(position.subtract(small_amount.yyx()), iTime).x ,            
        )
    )

def calculateNormal(position : vec3, iTime) : 
    #derivative xyy, yxy, yyx

    small_amount = 0.001 #slope
    return normalize(
        vec3(
            map_scene(position.add( vec3(small_amount,0,0)) , iTime).x - 
            map_scene(position.subtract( vec3(small_amount,0,0) ), iTime) .x, 

            map_scene(position.add(vec3(0,small_amount,0)), iTime).x - 
            map_scene(position.subtract(vec3(0,small_amount,0)), iTime).x , 
            
            map_scene(position.add(vec3(0,0,small_amount)), iTime).x - 
            map_scene(position.subtract(vec3(0,0,small_amount)), iTime).x ,            
        )
    )
#**kwargs **kwargs.get('x',none)
def castShadow(ro, rd, iTime) -> vec2:
    res : float = 1
    t = 0.01
    for i in range(100):
        pos : vec3 = ro.add(
            rd.multiply_by_float(t)
        ) #march  
        h : float= map_scene(pos, iTime).x
        res = min(res, 6* h/t)
        if h < 0.0001:
            break
        t+= h
        if t < 20: 
            break
    return clamp1(res,0,1)

def castRay (ray_origin, ray_direction, iTime) -> vec2:
    material : float = -1
    ray_march_step : float = 0.01
    for i in range(100):
        position_point : vec3 = ray_origin.add(
            ray_direction.multiply_by_float(ray_march_step)
        ) #march 
        hit : vec2 = map_scene(position_point, iTime)
        material = hit.y
        if hit.x < 0.001:
            break #inside
        ray_march_step += hit.x
        if ray_march_step > 20:
            break #too far outside
    if ray_march_step > 20:
        ray_march_step = -1
        material = -1
    return vec2(ray_march_step, material)

#https://www.youtube.com/watch?v=-pdSjBPH3zM  greek
#https://www.youtube.com/watch?v=Cfe5UQ-1L9Q
def mainImage(fragCoord : vec2, iResolution : vec3, iTime : float) -> vec3:
    
    p_pixel : vec2 = fragCoord.multiply_by_float(2).subtract(
        iResolution.xy()).divide_by_float(iResolution.y)
    #camera_ro_ray_origin : vec3 = vec3(
    #    1 * math.sin(iTime),
    #    0,
    #    2 * math.cos(iTime)
    #)  #rotate
    ta  : vec3= vec3(0,0.95,0)
    camera_ro_ray_origin : vec3 = vec3( 0,0.5,-2) #non-rotate versiion

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

    ray_march_step : vec2 = castRay(camera_ro_ray_origin, rd_ray_direction, iTime)

        
    if ray_march_step.y > 0: 
        #hit something
        t : float = ray_march_step.x
        position_point : vec3 = camera_ro_ray_origin.add(
            rd_ray_direction.multiply_by_float(t)
        )       
        normal : vec3= calculateNormal(position_point, iTime)



        material_base : vec3 = vec3(0.2,0.2,0.2) #base color

        #y contains material id
        if ray_march_step.y < 1.5:
            material_base = vec3(0.05,0.1,0.02)
        elif ray_march_step.y < 2.5:
            material_base = vec3(0.2,0.1,0.02)
        elif ray_march_step.y < 3.5:
            material_base = vec3(0.4,0.4,0.4)

        sun_direction : vec3 = normalize(vec3(0.8,0.4,0.2))
        sun_diffuse_light : float = clamp1 ( 
            vec3.dot(normal,sun_direction), 0, 1
        )
        #position we shading to light
        #sun_shadow = step1(
        #    castRay( 
        #        position_point.add(normal.multiply_by_float(0.001)), sun_direction, iTime
        #    ).y,0
        #)

        sun_shadow = castShadow(
                position_point.add(normal.multiply_by_float(0.001)), 
                sun_direction, 
                iTime
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
        #gamma = 2.2 ; 1/2.2 = 0.4545
        color_pixel = color_pixel.pow( vec3(0.4545,0.4545,0.4545)) #gamma correction brighter function
    return color_pixel

def getPPMString(image_width:int, image_height:int, array_rgb_values):
    file_content= f"P3\n{image_width} {image_height}\n255\n"
    for i in array_rgb_values:
        file_content+=f"{i[0]} {i[1]} {i[2]}\n"
    return file_content


def main():
    nx = IMAGE_WIDTH
    ny = IMAGE_HEIGHT
    iResolution = vec3(nx,ny,1)
    vec3_lower_left_corner=vec3(-2,-1,-1)
    vec3_horizontal=vec3(4,0,0)
    vec3_vertical=vec3(0,2,0)
    vec3_origin=vec3 (0,0,1)
    #cant use time.time since time will greatly elapse between frames?
    FRAME_PER_SECOND = 10 # 10 frames per second @ then 1 second = 10 frames
    for frame in range(1, 70 ): 
        array_rgb_values = []
        for j in range( (ny-1), -1, -1): #j=ny-1;j>=0;j--
            for i in range( 0, nx, 1): #i=0;i<nx;i++
                fragCoord : vec2 = vec2(i,j)
                time_elapsed_seconds = frame / FRAME_PER_SECOND # example at frame 20, 20/10 = 2 seconds elapsed
                fragColor = mainImage( fragCoord,iResolution, time_elapsed_seconds) #25 frames per second

                ir = clip( int(255* fragColor.r), 0, 255)
                ig = clip(int(255* fragColor.g), 0, 255)
                ib= clip(int(255* fragColor.b), 0, 255)
                array_rgb_values.append( (ir,ig,ib) )
                

        print(f"frame{frame}")
        ppm = open(f"ray-march-sphere-smin{frame:03}.ppm", "w")
        ppm.writelines( getPPMString(IMAGE_WIDTH, IMAGE_HEIGHT, array_rgb_values) )
        ppm.close()
        del ppm
        del array_rgb_values
        time.sleep(3) #cool down cpu
    
    print('done')    

main()




dot2 = lambda v  :  dot(v,v)
ndot = lambda  a , b : a.x*b.x - a.y*b.y; 

