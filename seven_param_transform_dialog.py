from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel
import os

FORM_CLASS, _ = uic.loadUiType(
    os.path.join(os.path.dirname(__file__), 'seven_param_transform_dialog.ui')
)

class SevenParamTransformDialog(QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # Optional: Add info label about new features
        #info_label = QLabel(
        #    "Features added:\n"
        #    "• Error ellipses (95% confidence)\n"
        #    "• Baarda data snooping for outliers\n"
        #    "• Residual visualization layers"
        #)
        #info_label.setStyleSheet("color: blue; font-size: 9px; padding: 5px;")
        #
        #layout = self.layout()
        #layout.insertWidget(2, info_label)  # Insert after author label