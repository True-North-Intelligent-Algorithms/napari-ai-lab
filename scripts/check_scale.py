from xml.etree import ElementTree as ET

from czifile import CziFile

for label, path in [
    (
        "original",
        r"c:\Users\bnort\work\ImageJ2022\tnia\napari-ai-lab\tests\test_images\vessels_large\Test lightsheet.czi",
    ),
    (
        "ds2",
        r"c:\Users\bnort\work\ImageJ2022\tnia\napari-ai-lab\tests\test_images\vessels_ds2\Test lightsheet ds2.czi",
    ),
]:
    print(label)
    with CziFile(path) as f:
        root = ET.fromstring(f.metadata())
        for ax in ("X", "Y", "Z"):
            node = root.find(".//Scaling/Items/Distance[@Id='%s']/Value" % ax)
            print(
                "  %s = %s"
                % (ax, node.text if node is not None else "not found")
            )
