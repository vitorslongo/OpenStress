import gmsh
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from matplotlib.tri import Triangulation

class Main:
    def __init__(self):
        # matplotlib.use('QtAgg')
        # TAREFA 5
        intern_radius = 30 / 1000
        outer_radius = 50 / 1000
        self.center = (0, 0, 0)
        self.p1 = (intern_radius, 0, 0)
        self.p2 = (outer_radius, 0, 0)
        self.p3 = (0, outer_radius, 0)
        self.p4 = (0, intern_radius, 0)
        self.nodes_in_curves = 20
        self.nodes_in_lines = 10
        self.intern_pressure = 10e6  # Pa
        self.E = 200e9
        self.nu = 0.3
        self.thickness = 1e-3 #m

        # TAREFA 4
        # self.L = 20e-3 #m
        # self.h = 10e-3 #m
        # self.t = 20e6 #Pa
        # self.thickness = 10e-3 #m
        # self.E = 2e9 #Pa
        # self.nu = 0.3

        # Options
        self.plot_K = 0

        # Pipeline
        gmsh_gui = 0
        self.generate_mesh(gmsh_gui=gmsh_gui, mesh_size=.2)
        print("Mesh generated")
        
        if gmsh_gui:
            return
        
        plot_mesh = 0
        if plot_mesh:
            self.plot_mesh()
            return
        
        self.assemble_global()
        print("Assembled matrixes")

        # quarto de circulo pressao interna
        # nodes_from_face_3 = [3, 18, 19, 20, 21, 22, 23, 24, 25, 0]
        nodes_from_intern_face = self.get_nodes_from_face(4)
        pressure_load = self.pressure_to_force_in_nodes(self.intern_pressure, nodes_from_intern_face)
        
        # print(pressure_load[0])
        loads = pressure_load
        dirichlet = [[self.get_nodes_from_face(1), 0, False, True], [self.get_nodes_from_face(3), 0, True, False]] 
        # dirichlet: [[nodes to apply, value, fix x, fix y]]

        # quarto de circulo força direita
        # loads = [[self.get_nodes_from_face(3), 1000, 0]]
        # dirichlet = [[self.get_nodes_from_face(1), 0, True, True]]

        # placa furada
        # loads = [[self.get_nodes_from_face(7), -10000, 0]]
        # dirichlet = [[self.get_nodes_from_face(8), 0, True, False]]

        # Tarefa 4
        # ================================
        # force = self.t*(self.thickness * self.h)
        # loads = [[self.get_nodes_from_face(2), force, 0]]
        # # dirichlet = [[self.get_nodes_from_face(4), 0, True, True]]
        # dirichlet = [[self.get_nodes_from_face(4), 0, True, False], [[0], 0, False, True]]


        self.apply_nodal_forces(loads)
        self.apply_dirichlet(dirichlet)


        print("Boundary conditions applied")
        
        print("Initating solution")
        self.U = self.solve()
        print("Solution calculated")

        print("Plotting results")
        scale_factor = 2e3
        # self.plot_displacement_results(scale_factor=scale_factor, loads=loads, dirichlet=dirichlet, forces=True)
        self.calculate_stress()
        self.plot_stresses(mesh=False, scale_factor=scale_factor)

    
    def plot_mesh(self):
        original_coords = self.nodal_coords[:, :2] 
        plotted_nodes = set()

        for _, n1, n2, n3 in self.conectivity:            
            x_original = [original_coords[n1, 0], original_coords[n2, 0], original_coords[n3, 0], original_coords[n1, 0]]
            y_original = [original_coords[n1, 1], original_coords[n2, 1], original_coords[n3, 1], original_coords[n1, 1]]

            plt.plot(x_original, y_original, linewidth=0.2, color="gray", zorder=0.1)
            plt.scatter(x_original, y_original, color="gray")
            if n1 not in plotted_nodes:
                plt.text(x_original[0], y_original[0], f"{n1}")
            if n2 not in plotted_nodes:
                plt.text(x_original[1], y_original[1], f"{n2}")
            if n3 not in plotted_nodes:
                plt.text(x_original[2], y_original[2], f"{n3}", horizontalalignment="center")
        
            plotted_nodes.add(n1)
            plotted_nodes.add(n2)
            plotted_nodes.add(n3)

        plt.show()

    def get_nodes_from_face(self, face):
        node_tags_gmsh = gmsh.model.mesh.getNodes(1, face)[0]
        node_tags = [node_gmsh - 1 for node_gmsh in node_tags_gmsh]
        
        point_entities = gmsh.model.getBoundary([(1, face)])
        for dim, tag in point_entities:
            node_tags.append(gmsh.model.mesh.getNodes(dim, tag)[0] - 1)
        return node_tags

    def plot_displacement_results(self, scale_factor=1., loads=list, dirichlet=list, forces=True):
        original_coords = self.nodal_coords[:, :2] 
        self.deformed_coords = original_coords + scale_factor * self.U
        plotted_nodes = set()

        for _, n1, n2, n3 in self.conectivity:
            
            x_deformed = [self.deformed_coords[n1, 0], self.deformed_coords[n2, 0], self.deformed_coords[n3, 0], self.deformed_coords[n1, 0]]
            y_deformed = [self.deformed_coords[n1, 1], self.deformed_coords[n2, 1], self.deformed_coords[n3, 1], self.deformed_coords[n1, 1]]
            
            x_original = [original_coords[n1, 0], original_coords[n2, 0], original_coords[n3, 0], original_coords[n1, 0]]
            y_original = [original_coords[n1, 1], original_coords[n2, 1], original_coords[n3, 1], original_coords[n1, 1]]

            plotted_nodes.add(n1)
            plotted_nodes.add(n2)
            plotted_nodes.add(n3)

            plt.plot(x_original, y_original, linewidth=0.2, color="gray", zorder=1)
            # plt.scatter(x_original, y_original, color="gray", zorder=1)
            if n1 not in plotted_nodes:
                plt.text(x_original[0], y_original[0], f"{n1}")
            if n2 not in plotted_nodes:
                plt.text(x_original[1], y_original[1], f"{n2}")
            if n3 not in plotted_nodes:
                plt.text(x_original[2], y_original[2], f"{n3}", horizontalalignment="center")


            plt.plot(x_deformed, y_deformed, linewidth=1, color="black", zorder=2)
            # plt.scatter(x_deformed, y_deformed, color="black", zorder=2)
            
            if forces:
                for nodes_applied, fx, fy in loads:
                    for node_idx in nodes_applied: 
                        x_coord = self.deformed_coords[node_idx, 0]
                        y_coord = self.deformed_coords[node_idx, 1]
                        
                        arrow_scale = 1e-5 
                        
                        plt.quiver(x_coord, y_coord, fx, fy, 
                                scale=1/arrow_scale,
                                angles='xy', scale_units='xy', 
                                color='red', width=0.005, headwidth=5, headlength=7, zorder=3)
            
            for nodes_applied, _, _, _ in dirichlet:
                for node_idx in nodes_applied:
                    x_coord = self.deformed_coords[node_idx, 0]
                    y_coord = self.deformed_coords[node_idx, 1]
                    plt.scatter(x_coord, y_coord, color="green", zorder=3)

        plt.show()

    def plot_stresses(self, scale_factor=1., field='von_mises', by='nodal', cmap='jet', mesh=False):
        elements = np.array([elem[1:] for elem in self.conectivity], dtype=int)
        original_coords = self.nodal_coords[:, :2] 
        self.deformed_coords = original_coords + scale_factor * self.U

        coords = np.asarray(self.deformed_coords)
        tri = Triangulation(coords[:, 0], coords[:, 1], elements)

        S = np.asarray(self.sigma_nodal)
        if field == 'sx':
            vals = S[:, 0]
            label = 'σ_x [Pa]'
        elif field == 'sy':
            vals = S[:, 1]
            label = 'σ_y [Pa]'
        elif field == 'txy':
            vals = S[:, 2]
            label = 'τ_xy [Pa]'
        else: 
            sx, sy, txy = S[:,0], S[:,1], S[:,2]
            vals = np.sqrt(sx*sx - sx*sy + sy*sy + 3*(txy**2))
            label = 'σ_von Mises [Pa]'

        plt.figure()
        plt.tripcolor(tri, vals, shading='gouraud', cmap=cmap)
        plt.colorbar(label=label)
        if mesh:
            for _, n1, n2, n3 in self.conectivity:
                x_deformed = [coords[n1,0], coords[n2,0], coords[n3,0], coords[n1,0]]
                y_deformed = [coords[n1,1], coords[n2,1], coords[n3,1], coords[n1,1]]
                plt.plot(x_deformed, y_deformed, linewidth=0.5, color="k", zorder=2)
                plt.scatter(coords[:,0], coords[:,1], color='k', s=6, zorder=3)

        plt.axis('equal')
        plt.gca().set_xlabel('X')
        plt.gca().set_ylabel('Y')
        plt.title(f'Tensão: {label} ({by})')
        plt.tight_layout()
        plt.show()

    def plot_matrix(self, matrix):
        c = plt.imshow(matrix, cmap="bwr")
        plt.colorbar(c)

        for (x, y), value in np.ndenumerate(matrix):
            plt.text(x, y, f"{value:.2e}", horizontalalignment="center", verticalalignment="center", color="green")
        plt.show()

    def generate_mesh(self, mesh_size=1, gmsh_gui=False):
        occ = gmsh.model.occ
        mesh = gmsh.model.mesh
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.RandomSeed", 1919)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)

        # =======================================================

        # p1 = occ.addPoint(*self.p1)
        # p2 = occ.addPoint(*self.p2)
        # p3 = occ.addPoint(*self.p3)
        # p4 = occ.addPoint(*self.p4)
        # center = occ.addPoint(*self.center)

        # l1 = occ.addLine(p1, p2)
        # l2 = occ.addCircleArc(p2, center, p3, center=True)
        # l3 = occ.addLine(p3, p4)
        # l4 = occ.addCircleArc(p4, center, p1, center=True)

        # wire = occ.addWire([l1, l2, l3, l4])
        # surface = occ.addPlaneSurface([wire])
        # occ.remove([(0, center)])
        # occ.synchronize()

        # mesh.setTransfiniteCurve(l1, self.nodes_in_lines)
        # mesh.setTransfiniteCurve(l3, self.nodes_in_lines)
        # mesh.setTransfiniteCurve(l4, self.nodes_in_curves)
        # mesh.setTransfiniteCurve(l2, self.nodes_in_curves)
        # mesh.setTransfiniteSurface(surface)

        # =======================================================

        # nodes = 10
        # occ.addRectangle(0, 0, 0, 1, 1)
        # for _, line in gmsh.model.getEntities(1):
        #     mesh.setTransfiniteCurve(line, nodes)
        # mesh.setTransfiniteSurface(1)

        # =======================================================

        rectangle = occ.addRectangle(0, 0, 0, 1, 1)
        circle = occ.addDisk(.5, .5, 0, .1, .1)
        occ.cut([(2, rectangle)], [(2, circle)])

        occ.synchronize()
        # field_id = gmsh.model.mesh.field.add("Distance")
        # gmsh.model.mesh.field.setNumbers(field_id, "EdgesList", [l[1] for l in gmsh.model.getEntities(1)])
        # gmsh.model.mesh.field.setNumber(field_id, "NumPointsPerCurve", 100)

        # threshold_field = gmsh.model.mesh.field.add("Threshold")
        # gmsh.model.mesh.field.setNumber(threshold_field, "InField", field_id)
        # gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", 0.01)  # smaller size near disk
        # gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", 0.05)  # default size
        # gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.09)  # inside this distance: SizeMin
        # gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.13)  # after this: SizeMax

        # gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        # =======================================================


        mesh.generate(2)
        mesh.removeDuplicateNodes()
        if gmsh_gui:
            gmsh.option.setNumber("Geometry.CurveLabels", 1)
            gmsh.fltk.run()
        self.structure_gmsh_data()

    def structure_gmsh_data(self):
        mesh = gmsh.model.mesh

        node_indexes_gmsh, coords, _ = mesh.getNodes(2, -1, True)
        
        self.number_of_nodes = len(node_indexes_gmsh)
        self.nodes_tags = node_indexes_gmsh - 1
        self.nodal_coords = coords.reshape(-1, 3)
        
        reorder_nodes_tags = np.argsort(self.nodes_tags)

        self.nodes_tags = self.nodes_tags[reorder_nodes_tags]
        self.nodal_coords = self.nodal_coords[reorder_nodes_tags]
        
        _, element_indexes_gmsh, element_nodes_gmsh = gmsh.model.mesh.getElements(2, -1)
        element_indexes_gmsh = element_indexes_gmsh[0]
        element_nodes_gmsh = element_nodes_gmsh[0]

        self.number_of_elements = len(element_indexes_gmsh)
        element_indexes = np.array(element_indexes_gmsh) - 1
        element_nodes = np.array(element_nodes_gmsh) - 1

        self.conectivity = np.zeros((self.number_of_elements, 4), dtype=int)
        self.conectivity[:, 0] = element_indexes
        self.conectivity[:, 1:] = element_nodes.reshape(-1, 3)

    def get_constitutive_matrix(self, EPT=False, EPD=False):
        E = self.E
        nu = self.nu
        if EPT:
            coef = E / (1 - nu**2)
            constitutive_matrix = coef * np.array(
                [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
            )
        elif EPD:
            raise Exception("Not implemented case")

        return constitutive_matrix

    def get_area_and_B(self, element_nodes, element_id):
        element_nodes_coords = []

        for node in element_nodes: # se dar erro aqui provavelmente tem um ponto solto na geometria
            node_coords = self.nodal_coords[node]
            element_nodes_coords.append(node_coords)

        x1, y1, z1 = element_nodes_coords[0]
        x2, y2, z2 = element_nodes_coords[1]
        x3, y3, z3 = element_nodes_coords[2]

        u = np.array([x2, y2, z2]) - np.array([x1, y1, z1])
        v = np.array([x3, y3, z3]) - np.array([x1, y1, z1])

        element_area = 0.5 * np.linalg.norm(np.cross(u, v))

        if abs(element_area) < 1e-10:
          raise ValueError(f"Collapsed element {element_id} with nodes {element_nodes} detected. Area: {element_area}.")

        x21 = x2 - x1
        y31 = y3 - y1

        y23 = y2 - y3
        y31 = y3 - y1
        y12 = y1 - y2
        x32 = x3 - x2
        x13 = x1 - x3

        B = (1.0 / (2.0 * element_area)) * np.array(
            [
                [y23, 0, y31, 0, y12, 0],
                [0, x32, 0, x13, 0, x21],
                [x32, y23, x13, y31, x21, y12],
            ]
        )

        return element_area, B

    def assemble_global(self):
        dofs = self.number_of_nodes * 2
        K = lil_matrix((dofs, dofs), dtype=np.float64)
        F = np.zeros((dofs), dtype=float)
        self.C = self.get_constitutive_matrix(EPT=True)

        for element_id, *element_nodes in self.conectivity:
            A, B = self.get_area_and_B(element_nodes, element_id)
            Ke = self.thickness * A * (B.T @ self.C @ B)

            dof_map = []
            for n in element_nodes:
                dof_map.extend([2*n, 2*n+1])
            for i in range(6):
                for j in range(6):
                    K[dof_map[i], dof_map[j]] += Ke[i, j]



        self.K = K
        self.F = F
        # self.plot_matrix(self.K)
   
    def apply_nodal_forces(self, loads=list):
        for nodes, fx, fy in loads:
            for node in nodes:
                dof = 2 * node  # Correção: remove o -1
                self.F[dof] += fx
                self.F[dof + 1] += fy

    def apply_dirichlet(self, dirichlet):
            for nodes_list, value, x, y in dirichlet: 
                for node_idx in nodes_list:    
                    dof_x = 2 * node_idx        
                    dof_y = 2 * node_idx + 1    
                    if x:
                        self.K[dof_x, :] = 0.
                        self.K[:, dof_x] = 0.
                        self.K[dof_x, dof_x] = 1.
                        self.F[dof_x] = value 
                    if y:
                        self.K[dof_y, :] = 0.
                        self.K[:, dof_y] = 0.
                        self.K[dof_y, dof_y] = 1.
                        self.F[dof_y] = value 
    
    def solve(self):
        self.K = self.K.tocsc()
        solver = splu(self.K)  # fatoração LU esparsa
        U = solver.solve(self.F)        
        U = U.reshape(-1, 2)
        return U

    def normal_2d(self, coords):
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        t = np.array([x2 - x1, y2 - y1])
        n = np.array([-t[1], t[0]])
        n = n / np.linalg.norm(n) 
        return n
        
    def pressure_to_force_in_nodes(self, pressure, nodes):
        load = []
        for i in range(0, len(nodes) - 1, 2):
            x1, y1 = self.nodal_coords[nodes[i], :2]
            x2, y2 = self.nodal_coords[nodes[i+1], :2]

            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            n = self.normal_2d([(x1, y1), (x2, y2)])
            force_per_node = pressure * length / 2
            load.append([[nodes[i]], n[0]*force_per_node, n[1]*force_per_node])
            load.append([[nodes[i+1]], n[0]*force_per_node, n[1]*force_per_node])
        return load

    def calculate_stress(self):
        self.stress = []
        self.sigma_nodal = np.zeros((self.number_of_nodes, 3))
        counts = np.zeros(self.number_of_nodes)

        for e, element in enumerate(self.conectivity):
            element_nodes = element[1:]
            A, B = self.get_area_and_B(element_nodes, element[0])

            U_element = self.U[element_nodes].reshape(6, 1)
            element_deformation = B @ U_element

            element_stress = self.C @ element_deformation # [sigma_x, sigma_y, tau_xy]

            
            for node in element_nodes:
                self.sigma_nodal[node, :] += element_stress[:, 0]
                counts[node] += 1

        self.sigma_nodal /= counts[:, None]  # média por nó

if __name__ == "__main__":
    Main()
