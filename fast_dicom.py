from fastai.vision import *

from collections import Counter

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

import h5py
import pydicom as dicom
import random
import string

from stl import mesh as stl_mesh


def max_agree(ls):
    """In an interable DT, select item with greatest frequency"""
    return max(Counter(ls).items(), key=lambda x: x[1])[0]

def anti_query(vec, axis):
    """Select everything from vector that is not specified axis."""
    query = np.ones_like(vec)
    query[axis] = 0
    return vec[query.astype(np.bool)]

def mapped(func, ls):
    """Evaluated mapping"""
    return list(map(func, ls))

def zipped(*args):
    """Evluated zipping"""
    return list(zip(*args))

def mkdir(path, **kwargs):
    """pathlib's mkdir but it returns the path"""
    path.mkdir(**kwargs)
    return path

def get_df(data_root):
    """Create a csv path and return the resulting dataframe.
       If exists than return that dataframe"""
    csv_path = data_root / "dcm_info.csv"

    if not csv_path.exists():
        df = pd.DataFrame(columns = ['scan_id', 'plane', 'index', 'z_depth', 'row_depth', 'col_depth'])
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        
    return df

class ImageDICOM(Image):
    """Class for storing a dicom image."""
    def __init__(self, px: torch.Tensor, plane: str, px_spacing: np.array):
        super().__init__(px)
        self.plane = plane
        self.px_spacing = px_spacing
        
    @classmethod
    def from_np(cls, arr:np.array, ind, dim, px_spacing, clip_low=-700, clip_high=300):
        plane = ['axial', 'coronal', 'saggital'][dim]
        arr_slice = np.take(arr, ind, dim)[None,...]
        clip_low = np.min(arr_slice)
        clip_high = np.max(arr_slice)
        arr_slice = np.clip(arr_slice, clip_low, clip_high)
        arr_slice[0,0] = clip_low
        arr_slice[0,1] = clip_high
        arr_slice -= clip_low
        arr_slice /= (clip_high - clip_low)
        t = tensor(arr_slice).float()
        return cls(t, plane, px_spacing)
    
    @classmethod
    def from_csv(cls, df, save_folder, study_name, slice_num, plane):
        row = df[df['scan_id'] == f'{study_name}_{plane}_{slice_num}.png']
        row = row.iloc[0].values
    
    def resize_px_spacing(self, px_spacing):
        plane_dim = ['axial', 'coronal', 'saggital'].index(self.plane)
        
        factor_change =  self.px_spacing / np.array(px_spacing)
        self.px_spacing = px_spacing
        factor_change = np.insert(anti_query(factor_change, plane_dim), 0, 1.)
        
        new_size = np.round((self.shape * factor_change))
        new_size = tuple([int(i) for i in new_size])
        return self.resize(tuple(new_size))

        
class ImageDICOMSegment(ImageDICOM, ImageSegment):
    @classmethod
    def from_np(cls, arr, ind, dim, px_spacing):
        plane = ['axial', 'coronal', 'saggital'][dim]
        arr_slice = np.take(arr, ind, dim)[None,...]
        t = tensor(arr_slice).float()
        return cls(t, plane, px_spacing)

    def save(self, fn:PathOrStr):
        "Save the image to `fn`."
        x = image2np(self.data).astype(np.uint8)
        PIL.Image.fromarray(x).save(fn)
        
        
class DcmGetterCIT():
    def get_scan_files(self, study_path):
        """Load in a study and clip the Houndsfield units."""
        ct_path = study_path / 'CT'
        return ct_path.ls()
    
    def get_mask(self, study_path):
        """Load CIT mask from segmentation file"""
        f = h5py.File(str(study_path / 'Segmentation.h5'), 'r')
        return np.ascontiguousarray(np.flip(np.array(f['mask']), 0))
        
class DcmGetterFH():
    def get_scan_files(self, study_path):
        """Load in a study and clip the Houndsfield units."""
        ct_path = study_path / 'Raw'
        return ct_path.ls()
    
    def get_mask(self, study_path):
        """Load CIT mask from segmentation file"""
        fp = study_path / 'Masks' / f'{study_path.stem}_masks.npy'
        arr = np.load(str(fp))
        arr = arr * np.array([1., 2, 3, 4]).reshape([1,4, 1, 1])
        arr = arr.max(axis=1)
        return np.flip(arr, [0,1]).astype(np.uint8)


class DcmProcessor():
    def __init__(self, path, save_folder, new_spacing=np.array([1.,1.,1.]), clip_low=-700, clip_high=300):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.new_spacing = new_spacing if new_spacing is not None else None
        self.path = path
        
        self.image_path, self.mask_path = self.get_folders(save_folder)
        self.df = pd.read_csv(path / "dcm_info.csv")
        
    def get_folders(self, save_folder):
        save_path = mkdir(self.path / save_folder, exist_ok=True)
        image_path = mkdir(save_path / "image", exist_ok=True)
        mask_path = mkdir(save_path / "mask", exist_ok=True) 
        return image_path, mask_path
    
    def get_scan(self, study_path):
        scan, px_spacing = self.proc_scan_files(self.get_scan_files(study_path))
        return scan, px_spacing

    def get_mri_mask(self, study_path):
    	mask, px_spacing = self.proc_scan_files(self.get_mask_files(study_path))
    	mask = np.where(mask != np.max(mask), 0, mask)
    	#mask = np.where(mask == np.max(mask), 256, mask)
    	return mask
        
        
    def proc_scan_files(self, scan_files):
        """Extensible function for dealing with """
        scan_files = (dicom.read_file(str(i)) for i in scan_files)

        #First slice is anterior
        scan_files = sorted(scan_files, key=lambda x: x.ImagePositionPatient[-1])

        # Convert to Houndsfield.
        slope, intercept = map(float, [scan_files[0].RescaleSlope, scan_files[0].RescaleIntercept])
        stack = [(i.pixel_array * slope) + intercept for i in scan_files][::-1]

        # Need spacing information for consistent conversion
        slice_dist = [abs(float(a.ImagePositionPatient[-1]) - float(b.ImagePositionPatient[-1])) for a, b in zip(scan_files[1:], scan_files[:-1])]
        slice_dist = max_agree(slice_dist)
        px_spacing = list(max_agree([tuple([float(i) for i in f.PixelSpacing]) for f in scan_files]))

        # Remove extraneous slices with wrong dimension
        max_dim = max_agree([i.shape for i in stack])
        stack = np.stack([i for i in stack if tuple(i.shape) == max_dim])
        # px_spacing should be row by column and not col by row... i assume
        #return np.clip(stack, self.clip_low, self.clip_high), np.array([slice_dist] + px_spacing[::-1])
        return stack, np.array([slice_dist] + px_spacing[::-1])
        

    def delete_study_files(self, study_path):
        for path in [self.image_path, self.mask_path]:
            [i.unlink() for i in path.ls() if study_path.stem in i.stem]

    def save_pair(self, fn, scan, mask, ind, plane, px_spacing):
        image = ImageDICOM.from_np(scan, ind, plane, px_spacing, clip_low=self.clip_low, clip_high=self.clip_high)
        mask = ImageDICOMSegment.from_np(mask, ind, plane, px_spacing)
        if self.new_spacing is not None:
            image.resize_px_spacing(self.new_spacing)
            mask.resize_px_spacing(self.new_spacing)
        image.save(self.image_path / fn)
        mask.save(self.mask_path / fn)
        

    def to_files(self, study_path):
        self.df = self.df[self.df['scan_id'] != study_path.stem]

        print(f"Creating data for {study_path.stem}")
        self.delete_study_files(study_path)

        scan, px_spacing = self.get_scan(study_path)
        mask = self.get_mri_mask(study_path)
        #mask = self.get_mask(study_path)

        slice_nums = sum([list(range(i)) for i in scan.shape], [])
        planes = sum([n_slices*[[0,1,2][ind]] for ind, n_slices in enumerate(scan.shape)], [])

        
        new_fn = lambda slice_num, plane: f'{study_path.stem}_{plane}_{slice_num}.png'
        save_func = lambda fn, slice_num, plane: self.save_pair(fn, scan, mask, slice_num, plane, px_spacing)
        
        df_rows = [[new_fn(slice_num, plane), plane, slice_num] + list(self.new_spacing) for slice_num, plane in zip(slice_nums, planes)]
        new_df = pd.DataFrame(df_rows)
        new_df.columns = self.df.columns
        self.df = self.df.append(new_df)
        
        for slice_num, plane in progress_bar(zipped(slice_nums, planes)):
            fn = new_fn(slice_num, plane)
            save_func(fn, slice_num, plane)
            
            
        self.df.to_csv(self.path / 'dcm_info.csv', index=False)

        
def create_dcm_class(name, get_cls, proc_cls):
    return type(name, (get_cls,), proc_cls.__dict__.copy())

DcmProcessorFH = create_dcm_class('DcmProcessorCIT', DcmGetterFH, DcmProcessor)
DcmProcessorCIT = create_dcm_class('DcmProcessorCIT', DcmGetterCIT, DcmProcessor)

def create_dataset(dcm_proc, study_paths):
    for study_path in study_paths:
        try:
            dcm_proc.to_files(study_path)
        except:
            print(f"Failed to save {study_path}")

            
# Code for evaluating trained model.

# TODO: Fix codes MP
                  
def dataset_val(path, val_study, axis, codes):
    split_func = lambda x: x.stem.split('_')[0] == val_study and x.stem.split('_')[1] == str(axis)
    
    # TODO: pLEASE FIX
    get_y_fn = lambda x: path / 'mask' / x.name
    
    db = (SegmentationItemList.from_folder(path)
          .filter_by_func(lambda x: 'mask' not in str(x))
          .split_by_valid_func(split_func))
    #Let's keep these in a nicer stack, thanks
    db.valid.items = np.array(sorted(db.valid.items, key=lambda x: int(x.stem.split('_')[-1])))
    orig_shape = db.valid[0].shape[-2:]
    close_shape = [i / 2 for i in orig_shape] if 512 in orig_shape else orig_shape
    close_shape = [int(round(i / 32)*32) for i in close_shape]
    db = (db.label_from_func(get_y_fn, classes=codes)
          .transform(2*[squish()], size=close_shape, resize_method=ResizeMethod.SQUISH, tfm_y=True,)
          .databunch(bs=2,  no_check=True)
          .normalize(imagenet_stats))
    return db, orig_shape
                  
class DicomEval():
    def __init__(self, learn, path):
        #Append miou callback for learn
        n_classes = learn.data.c
        self.learn = learn
        self.orig_data = learn.data
        
        self.path = path
        
        self.conf_mats = np.zeros((n_classes, n_classes))
        
    def one_study(self, study_name):
        res = []
        ys = None
        for ind, axis in enumerate(['0', '1', '2']):
            data, orig_shape = dataset_val(self.path, study_name, axis, codes=["background", "LA"])
            self.learn.data = data
            preds = self.learn.get_preds()
            xs = preds[0]
            xs = F.interpolate(xs, size=orig_shape)
            
            if ind == 1:
                xs = xs.permute(2, 1, 0, 3) #3,0
            elif ind == 2:
                xs = xs.permute(2,1,3,0)
            res.append(xs)
            if ys is None:
                ys = preds[1]
                ys = F.interpolate(ys.float(), size=orig_shape).long()
            
        self.learn.data = self.orig_data
        
        #Here we reshape the res tensors back.
        targ_shape = list(res[0].shape[1:])
        targ_shape.insert(0, res[1].shape[0])
        res = [resize_3d(i, size=tuple(targ_shape)) for i in res]
        targ_shape[1] = 1
        ys = resize_3d(ys, size=tuple(targ_shape))

 

        #res = sum(res).argmax(1)

        #res = res[2].argmax(1)
        return res, ys[:,0]
    
    def get_conf_mat(self, studies):
        self.conf_mats *= 0
        for study in progress_bar(studies):
            res, ys = self.one_study(study)
            self.conf_mats += confusion_matrix(res.flatten(), ys.flatten())
            
        return self.conf_mats
    
def resize_3d(t, size):
    """Convert a tensor back into original shape."""
    eq = tensor(t.shape) == tensor(size)
    if eq.all():
        return t
    else:
        ind = tensor(eq==0).nonzero().flatten()
        assert(len(ind) == 1)
        ind = ind[0]
        if ind != 0:
            return F.interpolate(t.float(), size=tuple(size[2:]))
        else:
            #Need to swap
            t = t.permute(2,1,0,3)
            t = F.interpolate(t.float(), size=tuple((size[0], size[-1])))
            return t.permute(2,1,0,3)
        

def save_3d_model(save_folder, study_name, dc_eval, spacing=[1.,1.,1.], classes=['background', 'RA',  'RV', 'LA', 'LV']):
    res, y = dc_eval.one_study(study_name)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    save_names = [save_folder / f"{study_name}_{i}.stl" for i in classes]

    for ind, save_name in enumerate(save_names):
        if "background" in save_name.stem:
            continue
        verts, faces, normals, values = measure.marching_cubes_lewiner( (res == ind).float().numpy()[::-1], 0, step_size=1, spacing=spacing)

        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).

        cube = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
        for ind, f in enumerate(faces):
            for j in range(3):
                cube.vectors[ind][j] = verts[f[j],::-1]

        cube.save(str(save_name))
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        mesh.set_facecolor('r')
        ax.add_collection3d(mesh)

    ax.set_xlim(0, 170)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 256)  # b = 10
    ax.set_zlim(0, 256)  # c = 16

    plt.tight_layout()
    plt.show()