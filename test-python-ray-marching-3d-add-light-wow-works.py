import math

class vec2:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = math.sqrt( 
            abs(x*x + y*y)
        )
        #self.xyy=vec3(x,y,y) #swizel recursion dont do
        #self.yxy=vec3(y,x,y) #swizel
        #self.yyx=vec3(y,y,x) #swizel

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
 
    def multiply(self,in_vec2):
        return vec2( self.x * in_vec2.x, self.y * in_vec2.y)
    
    def multiply_by_float(self, in_float):
        return vec2(self.x * in_float , self.y * in_float)

    def divide(self,in_vec2):
        return vec2( self.x / in_vec2.x, self.y / in_vec2.y)

    def divide_by_float(self, in_float):
        return vec2(self.x / in_float , self.y / in_float)




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
        self.xy = vec2(x,y)

    def add(self,in_vec3):
        return vec3( self.x + in_vec3.x, self.y+in_vec3.y, self.z+in_vec3.z)

    def add_by_float(self,in_float):
        return vec3( self.x + in_float, self.y+ in_float, self.z+ in_float)

    def subtract(self,in_vec3):
        return vec3( self.x - in_vec3.x, self.y-in_vec3.y, self.z-in_vec3.z)
 
    def multiply(self,in_vec3):
        return vec3( self.x * in_vec3.x, self.y * in_vec3.y, self.z * in_vec3.z)
    
    def multiply_by_float(self, in_float):
        return vec3(self.x * in_float , self.y * in_float, self.z * in_float)

    def divide(self,in_vec3):
        return vec3( self.x / in_vec3.x, self.y / in_vec3.y, self.z / in_vec3.z)

    def divide_by_float(self, in_float):
        return vec3(self.x / in_float , self.y / in_float, self.z / in_float)

    def __str__(self):
        return f"{self.x},{self.y},{self.z}"

    @staticmethod
    def unit_vector(in_vec3): #aka normalize
        return in_vec3.divide_by_float(in_vec3.length)
    
    @staticmethod
    def dot(v1_vec3,v2_vec3):
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

def clamp3(v3 : vec3 , lo, hi) -> vec3:
    #lo_v3 : vec3 = vec3(lo,lo,lo)
    #hi_v3 : vec3 = vec3(hi,hi,hi)
    return vec3 (
        clip(v3.x, lo, hi),
        clip(v3.y, lo, hi),
        clip(v3.z, lo, hi)
    )

def map_scene(p3):
    return min(
        distance3(p3, vec3(-1,0,-5))-1,
        distance3(p3, vec3(2,0,-3))-1,
        distance3(p3, vec3(-2,0,-2))-1,
        p3.y + 1
    )

def calcNormal(p3):
    e2 = vec2(1,-1) * 0.0005
    return normalize(
        e2.xyy * map_scene(p3 + e2.xyy) +
        e2.yyx * map_scene(p3 + e2.yyz) +
        e2.yxy * map_scene(p3 + e2.yxy) +
        e2.xxx * map_scene(p3 + e2.xxx)

    )

def mainImage(fragCoord,iResolution):

    p = fragCoord.multiply_by_float(2).subtract(iResolution.xy).divide_by_float(iResolution.y)
    d = sdCircle( p , 0.5)

    col = vec3(1,1,1).subtract(
        vec3(0.1,0.4,0.7).multiply_by_float ( sign(d) )
    ) 


    return col



#main()
#https://www.youtube.com/watch?v=PGtv-dBi2wE

MAX_STEPS : int = 100
MAX_DIST : float = 100
SURF_DIST : float = 0.01
IMAGE_WIDTH : int = 512
IMAGE_HEIGHT : int = 512


#http://glprogramming.com/red/appendixf.html
#w is forth coordinate
def GetDist(p : vec3) -> float:
    sphere : vec3 = vec3(0,1,6) #w=1=radius
    sphereDist : float = (p.subtract(sphere)).length - 1
    planeDist = p.y
    d : float = min(sphereDist, planeDist)
    return d


def RayMarch(ro : vec3, rd : vec3) -> float:
    dO : float = 0 #distance origin
    for i in range(0,MAX_STEPS):
        p : vec3 = ro.add( rd.multiply_by_float(dO))
        dS : float = GetDist(p) #distance scene
        dO += dS
        if (dS < SURF_DIST or dO > MAX_DIST):
            break
    return dO

def GetNormal(p : vec3) -> vec3:
    d : float= GetDist(p)
    e : vec2 = vec2(0.01,0)
    n : vec3 = vec3(
        d- GetDist(p.subtract(vec3(e.x,e.y,e.y))),   
        d- GetDist(p.subtract(vec3(e.y,e.x,e.y))),   
        d- GetDist(p.subtract(vec3(e.y,e.y,e.x)))       

    )
    return normalize(n)

def GetLight(p : vec3) -> float:
    lightPos : vec3 = vec3(0,5,6)
    l : vec3= normalize(lightPos.subtract(p))
    n : vec3 = GetNormal(p)
    dif : float = clamp1 ( dot(n,l), 0, 1)
    d : float = RayMarch (
        p.add(
            n.multiply_by_float(SURF_DIST).multiply_by_float(2)
        ),
        l
    )
    if d < lightPos.subtract(p).length :
        dif = dif * 0.1
    return dif

#16:43
def glsl(fragCoord : vec2, iResolution : vec3) -> vec3:
    uv : vec2 = (fragCoord.subtract(iResolution.xy.multiply_by_float(0.5))).divide_by_float(iResolution.y)
    col : vec3 = vec3(0,1,0)
    rO : vec3 = vec3(0,1,0)
    rd : vec3 = normalize(vec3(uv.x, uv.y, 1 ) )
    
    d : float = RayMarch(rO,rd)
    p : vec3= rO.add( rd.multiply_by_float(d))
    dif : float = GetLight(p)
    #d = d / 6
    #col = vec3(d,d,d)
    #col = GetNormal(p)
    col = vec3(dif,dif,dif)
    return col

def main_march():
    nx = IMAGE_WIDTH
    ny = IMAGE_HEIGHT
    iResolution = vec3(nx,ny,1)
    file_content= f"P3\n{nx} {ny}\n255\n"
    vec3_lower_left_corner=vec3(-2,-1,-1)
    vec3_horizontal=vec3(4,0,0)
    vec3_vertical=vec3(0,2,0)
    vec3_origin=vec3 (0,0,1)

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
            fragColor = glsl( fragCoord,iResolution)
            ir = int(255.99* fragColor.r)
            ig = int(255.99* fragColor.g)
            ib= int(255.99* fragColor.b)
            file_content+=f"{ir} {ig} {ib}\n"

    #print(file_content)
    ppm = open("./testppm-glsl-art-code-light.ppm", "w")
    ppm.writelines(file_content)
    ppm.close()
    print('done')    

main_march()
