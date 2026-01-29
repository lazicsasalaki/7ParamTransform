import numpy as np
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, 
    QgsPointXY, QgsField, QgsFields, QgsWkbTypes,
    QgsMarkerSymbol, QgsFillSymbol, QgsLineSymbol,
    QgsSingleSymbolRenderer, QgsCategorizedSymbolRenderer,
    QgsRendererCategory, QgsSymbol, QgsSimpleMarkerSymbolLayer,
    QgsSimpleFillSymbolLayer, QgsSimpleLineSymbolLayer
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from .seven_param_transform_dialog import SevenParamTransformDialog
from scipy.stats import chi2, norm
import math
import os

class SevenParamTransform:

    def __init__(self, iface):
        self.iface = iface
        self.points_layer = None
        self.residual_layer = None
        self.ellipse_layer = None

    def initGui(self):
        self.action = QAction(
            "LS – 7-Param Helmert",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(
            "7-Param Transform",
            self.action
        )

    def unload(self):
        self.iface.removePluginMenu(
            "7-Param Transform",
            self.action
        )
        # Clean up layers
        self.remove_layers()

    def remove_layers(self):
        """Remove created layers from QGIS"""
        layer_names = ['Helmert_Residuals', 'Error_Ellipses_95%', 'Baarda_Outliers']
        for name in layer_names:
            layer = QgsProject.instance().mapLayersByName(name)
            if layer:
                QgsProject.instance().removeMapLayer(layer[0].id())

    def run(self):
        dlg = SevenParamTransformDialog(self.iface.mainWindow())

        dlg.pointsBrowseBtn.clicked.connect(
            lambda: self.browse_file(dlg, dlg.pointsFileEdit)
        )

        dlg.runButton.clicked.connect(
            lambda: self.compute_ls(dlg.pointsFileEdit.text())
        )

        dlg.exec_()

    def browse_file(self, dlg, line_edit):
        fname, _ = QFileDialog.getOpenFileName(
            dlg,
            "Select points.txt",
            "",
            "Text files (*.txt)"
        )
        if fname:
            line_edit.setText(fname)

    # ================= LS COMPUTATION WITH ERROR ELLIPSES =================
    def compute_ls(self, filename):
        try:
            A, L = [], []
            ids = []
            coords = []  # Store original coordinates for visualization

            with open(filename, "r", encoding="utf-8") as f:
                next(f)  # header

                for line in f:
                    if not line.strip():
                        continue

                    p = line.split()
                    ids.append(p[0])

                    X1 = float(p[1])
                    Y1 = float(p[2])
                    Z1 = float(p[3])
                    X2 = float(p[4])
                    Y2 = float(p[5])
                    Z2 = float(p[6])

                    coords.append((X1, Y1, X2, Y2))

                    A.extend([
                        [1, 0, 0, 0,  Z1, -Y1,  X1],
                        [0, 1, 0, -Z1, 0,  X1,  Y1],
                        [0, 0, 1,  Y1, -X1, 0,  Z1]
                    ])

                    L.extend([X2-X1, Y2-Y1, Z2-Z1])

            A = np.array(A, dtype=float)
            L = np.array(L, dtype=float)

            # Least squares solution
            N = A.T @ A
            U = A.T @ L
            p = np.linalg.solve(N, U)

            v = A @ p - L
            dof = len(L) - len(p)
            sigma0 = np.sqrt((v.T @ v) / dof)

            # --- CHI-SQUARE TEST ---
            alpha = 0.05
            chi2_val = (v.T @ v) / (sigma0**2)
            chi2_min = chi2.ppf(alpha/2, dof)
            chi2_max = chi2.ppf(1 - alpha/2, dof)
            chi_pass = chi2_min <= chi2_val <= chi2_max

            # Covariance matrix of parameters
            Cov_p = sigma0**2 * np.linalg.inv(N)

            # --- ERROR ELLIPSES AND BAARDA TEST ---
            residuals, ellipses, baarda_results = self.compute_error_analysis(A, v, sigma0, ids, coords)

            # Print results
            self.print_results(p, sigma0, Cov_p, v, ids, dof, chi2_val, chi2_min, chi2_max, chi_pass, baarda_results)

            # Create visualization layers
            self.create_residual_layer(ids, coords, residuals, baarda_results)
            self.create_ellipse_layer(ids, coords, ellipses)

            QMessageBox.information(None, "Success", 
                                   "7-parameter transformation completed!\n"
                                   f"Global Chi-square test: {'PASS' if chi_pass else 'FAIL'}\n"
                                   f"Outliers detected: {sum(1 for b in baarda_results if b['is_outlier'])}\n"
                                   f"Error ellipses created with 5x scaling for visualization.")

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Computation failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def compute_error_analysis(self, A, v, sigma0, ids, coords):
        """Compute error ellipses and perform Baarda's data snooping"""
        n_points = len(ids)
        residuals = []
        ellipses = []
        baarda_results = []

        # For each point, compute covariance matrix of residuals
        for i in range(n_points):
            # Get design matrix for this point (3x7)
            Ai = A[3*i:3*i+3, :]
            
            # Cofactor matrix for this point's residuals
            Q_vv_i = np.eye(3) - Ai @ np.linalg.pinv(A.T @ A) @ Ai.T
            
            # Standardized residuals for Baarda test
            w_x = v[3*i] / (sigma0 * np.sqrt(Q_vv_i[0, 0]))
            w_y = v[3*i+1] / (sigma0 * np.sqrt(Q_vv_i[1, 1]))
            w_z = v[3*i+2] / (sigma0 * np.sqrt(Q_vv_i[2, 2]))
            
            # Maximum standardized residual (for this point)
            w_max = max(abs(w_x), abs(w_y), abs(w_z))
            
            # Baarda test (α = 0.05, two-tailed)
            alpha_local = 0.05
            n_observations = len(v)
            quantile = norm.ppf(1 - alpha_local/(2*n_observations))
            
            is_outlier = w_max > quantile
            
            # Store Baarda results
            baarda_results.append({
                'id': ids[i],
                'w_x': w_x,
                'w_y': w_y,
                'w_z': w_z,
                'w_max': w_max,
                'quantile': quantile,
                'is_outlier': is_outlier
            })
            
            # Store residuals
            residuals.append({
                'id': ids[i],
                'vx': v[3*i],
                'vy': v[3*i+1],
                'vz': v[3*i+2],
                'sigma_vx': sigma0 * np.sqrt(Q_vv_i[0, 0]),
                'sigma_vy': sigma0 * np.sqrt(Q_vv_i[1, 1]),
                'sigma_vz': sigma0 * np.sqrt(Q_vv_i[2, 2])
            })
            
            # Compute error ellipse parameters (horizontal only - X,Y)
            # Extract 2x2 covariance for XY
            Q_xy = Q_vv_i[:2, :2] * (sigma0**2)
            
            # Eigenvalues and eigenvectors for error ellipse
            eigenvalues, eigenvectors = np.linalg.eig(Q_xy)
            
            # Semi-major and semi-minor axes (95% confidence)
            confidence = 2.4477  # sqrt(chi2(2, 0.95))
            a = confidence * np.sqrt(max(eigenvalues))
            b = confidence * np.sqrt(min(eigenvalues))
            
            # ORIENTATION - fix for QGIS (clockwise from North)
            # Calculate angle from eigenvectors
            dx = eigenvectors[0, 0]  # eigenvector x-component for major axis
            dy = eigenvectors[1, 0]  # eigenvector y-component for major axis
            
            # Convert to angle from North (clockwise) in degrees for QGIS
            angle_rad = np.arctan2(dx, dy)  # Note: swapped dx/dy for North reference
            orientation = np.degrees(angle_rad)
            
            # Make sure orientation is between 0 and 360
            orientation = orientation % 360
            
            # APPLY 5x SCALING FOR BETTER VISIBILITY
            scale_factor = 5.0
            a_scaled = a * scale_factor
            b_scaled = b * scale_factor
            
            # Store ellipse parameters
            ellipses.append({
                'id': ids[i],
                'center_x': coords[i][0],  # Source X
                'center_y': coords[i][1],  # Source Y
                'semi_major': a,
                'semi_minor': b,
                'semi_major_scaled': a_scaled,
                'semi_minor_scaled': b_scaled,
                'orientation': orientation,
                'scale_factor': scale_factor
            })

        return residuals, ellipses, baarda_results

    def print_results(self, p, sigma0, Cov_p, v, ids, dof, chi2_val, chi2_min, chi2_max, chi_pass, baarda_results):
        """Print comprehensive results"""
        print("\n" + "="*50)
        print("LS 7-PARAMETER HELMERT TRANSFORMATION")
        print("="*50)
        
        print(f"\n--- TRANSFORMATION PARAMETERS ---")
        print(f"Tx = {p[0]:.4f} m")
        print(f"Ty = {p[1]:.4f} m")
        print(f"Tz = {p[2]:.4f} m")
        print(f"Rx = {p[3]*206264.806:.6f} arcsec")
        print(f"Ry = {p[4]*206264.806:.6f} arcsec")
        print(f"Rz = {p[5]*206264.806:.6f} arcsec")
        print(f"Scale = {p[6]*1e6:.4f} ppm")

        print(f"\n--- APOSTERIORI STANDARD DEVIATION ---")
        print(f"sigma0 = {sigma0:.6f} m")

        print(f"\n--- PARAMETER VARIANCES ---")
        for i in range(7):
            print(f"σ²[{i+1}] = {Cov_p[i,i]:.6e}")

        print(f"\n--- GLOBAL χ² TEST (α=0.05) ---")
        print(f"Degrees of freedom = {dof}")
        print(f"χ² = {chi2_val:.4f}")
        print(f"χ²_min = {chi2_min:.4f}")
        print(f"χ²_max = {chi2_max:.4f}")
        print(f"Result : {'PASS ✅' if chi_pass else 'FAIL ❌'}")

        print(f"\n--- BAARDA DATA SNOPING (α=0.05) ---")
        print(f"{'ID':<10} {'w_x':<10} {'w_y':<10} {'w_z':<10} {'w_max':<10} {'Quantile':<10} {'Status'}")
        print("-" * 80)
        for result in baarda_results:
            status = "OUTLIER ❌" if result['is_outlier'] else "OK ✅"
            print(f"{result['id']:<10} {result['w_x']:9.3f} {result['w_y']:9.3f} "
                  f"{result['w_z']:9.3f} {result['w_max']:9.3f} {result['quantile']:9.3f}  {status}")

        print(f"\n--- RESIDUALS PER POINT ---")
        for i, pid in enumerate(ids):
            vx = v[3*i]
            vy = v[3*i+1]
            vz = v[3*i+2]
            print(f"{pid}: vx={vx:7.4f} vy={vy:7.4f} vz={vz:7.4f}")

    def create_residual_layer(self, ids, coords, residuals, baarda_results):
        """Create vector layer for residuals visualization"""
        # Remove existing layer
        self.remove_layers()

        # Create memory layer for residuals
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "Helmert_Residuals", "memory")
        provider = layer.dataProvider()
        
        # Add fields
        fields = QgsFields()
        fields.append(QgsField("ID", QVariant.String))
        fields.append(QgsField("vX", QVariant.Double))
        fields.append(QgsField("vY", QVariant.Double))
        fields.append(QgsField("vZ", QVariant.Double))
        fields.append(QgsField("w_max", QVariant.Double))
        fields.append(QgsField("Outlier", QVariant.String))
        fields.append(QgsField("Sigma_vX", QVariant.Double))
        fields.append(QgsField("Sigma_vY", QVariant.Double))
        fields.append(QgsField("Sigma_vZ", QVariant.Double))
        provider.addAttributes(fields)
        layer.updateFields()

        # Add features
        for i, (pid, coord, res, baarda) in enumerate(zip(ids, coords, residuals, baarda_results)):
            feat = QgsFeature()
            feat.setFields(fields)
            
            # Use source coordinates for point location
            point = QgsPointXY(coord[0], coord[1])
            feat.setGeometry(QgsGeometry.fromPointXY(point))
            
            feat.setAttributes([
                pid,
                res['vx'],
                res['vy'],
                res['vz'],
                baarda['w_max'],
                "YES" if baarda['is_outlier'] else "NO",
                res['sigma_vx'],
                res['sigma_vy'],
                res['sigma_vz']
            ])
            
            provider.addFeature(feat)

        layer.updateExtents()
        
        # Apply styling
        self.style_residual_layer(layer)
        
        QgsProject.instance().addMapLayer(layer)
        self.residual_layer = layer

    def create_ellipse_layer(self, ids, coords, ellipses):
        """Create polygon layer for error ellipses"""
        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", "Error_Ellipses_95%", "memory")
        provider = layer.dataProvider()
        
        # Add fields
        fields = QgsFields()
        fields.append(QgsField("ID", QVariant.String))
        fields.append(QgsField("SemiMajor", QVariant.Double))
        fields.append(QgsField("SemiMinor", QVariant.Double))
        fields.append(QgsField("Orientation", QVariant.Double))
        fields.append(QgsField("ScaleFactor", QVariant.Double))
        fields.append(QgsField("SemiMajor_orig", QVariant.Double))
        fields.append(QgsField("SemiMinor_orig", QVariant.Double))
        provider.addAttributes(fields)
        layer.updateFields()

        # Add features (ellipses as polygons)
        for ellipse in ellipses:
            feat = QgsFeature()
            feat.setFields(fields)
            
            # Create ellipse polygon USING SCALED DIMENSIONS FOR VISIBILITY
            ellipse_poly = self.create_ellipse_polygon(
                ellipse['center_x'], ellipse['center_y'],
                ellipse['semi_major_scaled'], ellipse['semi_minor_scaled'],
                ellipse['orientation']
            )
            
            feat.setGeometry(ellipse_poly)
            feat.setAttributes([
                ellipse['id'],
                ellipse['semi_major_scaled'],  # Scaled for display
                ellipse['semi_minor_scaled'],  # Scaled for display
                ellipse['orientation'],
                ellipse['scale_factor'],
                ellipse['semi_major'],  # Original
                ellipse['semi_minor']   # Original
            ])
            
            provider.addFeature(feat)

        layer.updateExtents()
        
        # Apply styling with orange border
        self.style_ellipse_layer(layer)
        
        QgsProject.instance().addMapLayer(layer)
        self.ellipse_layer = layer
        
        # Zoom to layer
        self.iface.mapCanvas().setExtent(layer.extent())
        self.iface.mapCanvas().refresh()

    def create_ellipse_polygon(self, center_x, center_y, a, b, orientation):
        """Create polygon geometry for error ellipse"""
        points = []
        n_points = 72  # Increased for smoother ellipse
        
        # Convert orientation to radians (QGIS uses degrees clockwise from North)
        theta = np.radians(-orientation)  # Negative for clockwise
        
        for i in range(n_points + 1):
            angle = 2 * np.pi * i / n_points
            
            # Parametric equation of ellipse
            x_ell = a * np.cos(angle)
            y_ell = b * np.sin(angle)
            
            # Rotate ellipse
            x_rot = x_ell * np.cos(theta) - y_ell * np.sin(theta)
            y_rot = x_ell * np.sin(theta) + y_ell * np.cos(theta)
            
            # Translate to center
            x = center_x + x_rot
            y = center_y + y_rot
            
            points.append(QgsPointXY(x, y))
        
        # Close the polygon
        points.append(points[0])
        
        return QgsGeometry.fromPolygonXY([points])

    def style_residual_layer(self, layer):
        """Apply styling to residuals layer"""
        # Simple styling without QgsRendererCategory to avoid import issues
        symbol = QgsMarkerSymbol.createSimple({
            'name': 'circle',
            'color': '255,0,0',  # Red for all points initially
            'size': '3',
            'outline_color': '100,0,0'
        })
        
        renderer = QgsSingleSymbolRenderer(symbol)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def style_ellipse_layer(self, layer):
        """Apply styling to ellipse layer - orange border"""
        # Create simple orange border with transparent fill
        symbol = QgsFillSymbol.createSimple({
            'color': '255,165,0,40',  # Semi-transparent orange fill
            'color_border': '255,69,0',  # Dark orange border
            'width_border': '0.8'
        })
        
        # Apply renderer
        renderer = QgsSingleSymbolRenderer(symbol)
        layer.setRenderer(renderer)
        layer.triggerRepaint()