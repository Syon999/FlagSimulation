import taichi as ti
from PIL import Image # IMPORTANTE: ejecutar en consola uv add pillow, de esta manera añadimos el modulo de pillow que nos permite ver la textura
import numpy as np

ti.init(arch=ti.gpu)

# Parametros de la pieza de la tela 1x1
n = 128                    # numero de particulas
quad_size = 1.0 / (n - 1) # distancia entre particulas en reposo
m = 1 / (n * n)            # la masa ||||||SIEMPRE PONER PARENTESIS AL DENOMINADOR|||||||||

cloth_corner = ti.Vector([-0.5, 0.7, -0.5])

# pos y velocidad de las particulas
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
a = ti.Vector.field(3, dtype=float, shape=(n, n))

# estructura de datos para dibujar la malla
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
num_triangulos = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangulos * 3)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# convertir la imagen para trabajar con ella
img = Image.open("cage.png").convert("RGB")
img = img.resize((n, n))
img = np.asarray(img).astype(np.float32) / 255.0
texture = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
texture.from_numpy(img)

@ti.kernel
def calculate_indices():
    for i, j in ti.ndrange(n - 1, n - 1): # Recorro los cuadrados
        quad_id = i * (n - 1) + j
        # Rellenar indices 1er triangulo
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # Rellenar indices 2do triangulo
        indices[quad_id * 6 + 3] = i * n + (j + 1)
        indices[quad_id * 6 + 4] = (i + 1) * n + j
        indices[quad_id * 6 + 5] = (i + 1) * n + (j + 1)

@ti.kernel
def calculate_colors():
    for i, j in x: # ti,ndrange(n, n):
        # TEXTURA PUNTO EXTRA: dado que no podemos directamente asignar una imagen a un campo textura ya predefinido, lo que hacemos es
        #asignarle pixel a pixel a la tela los colores de la imagen, de manera que la tela resultante tenga la imagen como textura
        colors[i * n + j] = texture[i, j]

# Parametros simulación
dt = 0.0003                     # paso de tiempo (55 pasos por frame a 60fps)
g = ti.Vector([0, -9.8, 0])    # gravedad

# Constantes de los muelles según el enunciado: ke=2n, kf=2n/2, kd=2n/sqrt(2), dm=2n*1e-5
K_estructural = 2 * n           # ke = 2n = 256  →  rigidez muelles estructurales
K_flexion     = 2 * n / 2      # kf = 2n/2 = 128  →  rigidez muelles de flexión
K_deformacion = 2 * n / ti.sqrt(2.0)  # kd = 2n/√2 ≈ 181.01  →  rigidez muelles de deformacion (shear)
K_damping     = 2 * n * 1e-5   # dm = 2n×10⁻⁵ = 0.00256  →  constante de amortiguamiento de los muelles
d_aire        = 1 / (n * n)    # d = 1/n² ≈ 0.0000610  →  constante resistencia al aire

@ti.kernel
def step(): # Symplectic Euler
    for I in ti.grouped(x):
        # Fuerzas
        # Fuerza de la gravedad
        F_g = m * g
        #
        F_amortiguamiento = ti.Vector([0.0, 0.0, 0.0])

        # Fuerza muelles estructurales
        # enlace (i,j)&(i+1,j) y (i,j)&(i,j+1)  →  vecinos a distancia 1 en horizontal/vertical
        F_estructural = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                P_ij = x[J] - x[I] # vector del muelle de i hacia j
                dist = P_ij.norm() # longitud actual del muelle
                L0 = quad_size # longitud en reposo
                if dist > 1e-6:
                    # Ley de Hooke: F = K*(|Pij| - L0) * Pij/|Pij|
                    F_estructural += K_estructural * (dist - L0) * (P_ij / dist)
                    # Amortiguamiento: F = dm * (v_ij · P̂_ij) * P̂_ij
                    # Se opone únicamente a los cambios en la longitud del muelle
                    v_ij = v[J] - v[I]
                    F_amortiguamiento += K_damping * v_ij.dot(P_ij / dist) * (P_ij / dist)

        # Fuerza muelles de deformacion (shear/corte)
        # enlace (i,j)&(i+1,j+1) y (i+1,j)&(i,j+1)  →  vecinos diagonales
        F_deformacion = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                P_ij = x[J] - x[I]
                dist = P_ij.norm()
                L0 = quad_size * ti.sqrt(2.0) # longitud en reposo diagonal
                if dist > 1e-6:
                    F_deformacion += K_deformacion * (dist - L0) * (P_ij / dist)
                    v_ij = v[J] - v[I]
                    F_amortiguamiento += K_damping * v_ij.dot(P_ij / dist) * (P_ij / dist)

        # Fuerza muelles de flexion
        # enlace (i,j)&(i+2,j) y (i,j)&(i,j+2)  →  vecinos a distancia 2, se "saltan" una partícula
        F_flexion = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-2, 0), (2, 0), (0, -2), (0, 2)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                P_ij = x[J] - x[I]
                dist = P_ij.norm()
                L0 = quad_size * 2.0 # longitud en reposo (2 veces quad_size)
                if dist > 1e-6:
                    F_flexion += K_flexion * (dist - L0) * (P_ij / dist)
                    v_ij = v[J] - v[I]
                    F_amortiguamiento += K_damping * v_ij.dot(P_ij / dist) * (P_ij / dist)

        # Fuerza de resistencia al aire: F_aire = -d * v
        F_aire = -d_aire * v[I]

        F = F_g + F_estructural + F_deformacion + F_flexion + F_amortiguamiento + F_aire
        # Aceleración
        a[I] = F / m

    # Actualización velocidad y posición
    for I in ti.grouped(x): # I = i, j
        v[I] += a[I] * dt

    # Esquinas fijas: aceleración nula, velocidad nula, posición constante
    v[0, 0] = [0, 0, 0]
    v[0, n - 1] = [0, 0, 0]
    # Actualización de la posición
    for i, j in x:

        x[i, j] += v[i, j] * dt

    # Mantener esquinas fijas en su posición original
    x[0, 0] = [cloth_corner.x, cloth_corner.y, cloth_corner.z]
    x[0, n - 1] = [cloth_corner.x, cloth_corner.y, cloth_corner.z + (n - 1) * quad_size]


@ti.kernel
def update_vertices():
    for i, j in x:
        vertices[i * n + j] = x[i, j]

@ti.kernel
def initialize_cloth():
    for i, j in x:
        x[i, j] = [
            cloth_corner.x + i * quad_size,
            cloth_corner.y,
            cloth_corner.z + j * quad_size
        ]

        v[i, j] = [0, 0, 0]

def init():
    initialize_cloth()
    calculate_indices()
    calculate_colors()
    update_vertices()  # inicializar vertices antes del primer frame para que la malla sea visible


def main():
    window = ti.ui.Window("Simulacion de tela", (512, 512), fps_limit=60)
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(1.3, 0.5, 0.0)
    camera.lookat(-0.5, 0.3, 0.0)
    camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)

    init()

    while window.running:
        # Simulación: 55 pasos por frame (dt=0.0003, 60fps → 1/60//0.0003 = 55)
        for _ in range(int(1 / 60 // dt)):
            step()

        # Renderizado
        update_vertices()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light((0., 1., 2.), (1., 1., 1.))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(vertices, indices, per_vertex_color=colors, show_wireframe=False)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()