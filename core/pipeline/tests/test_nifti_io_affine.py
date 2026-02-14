import numpy as np
import nibabel as nib

from src.preprocess.utils_io import save_nifti


def test_save_nifti_preserves_affine_and_header(tmp_path):
    data = np.zeros((5, 6, 7), dtype=np.float32)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ref = nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.float32), affine=affine)

    out_path = tmp_path / "out.nii.gz"
    save_nifti(data, ref, str(out_path))

    img = nib.load(str(out_path))
    assert img.shape == (5, 6, 7)
    assert np.allclose(img.affine, affine)
