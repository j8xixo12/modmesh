from .. import core
from .. import view as mh_view
from .. import apputil

from PUI.PySide6 import *
from PySide6 import Qt

class MeshViewer(Application):
    def __init__(self):
        super().__init__()
        self.open_file_path = None

    def content(self):
        with Window(title="Mesh viewer", size=(640,480)):
            print(type(self.ui))
            self.w_mh_viewer = mh_view.R3DWidget(self.ui)
            with MenuBar():
                with Menu("File"):
                    MenuAction("Open").trigger(self.open_file_cb)

    def open_file_cb(self):
        self.open_file_path = OpenFile()
        data = open(self.open_file_path, 'rb').read()
        gm = core.Gmsh(data)
        mh = gm.to_block()
        print(type(mh))
        self.w_mh_viewer.updateMesh(mh)
        self.w_mh_viewer.show()

def load_app():
    app = MeshViewer()
    app.run()
