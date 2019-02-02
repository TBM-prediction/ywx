from utils.imports import *
from fastai.tabular.data import OptTabTfms

class MultiDeptTabularDataBunch(DataBunch):
    "Create a `DataBunch` suitable for tabular data."

    @classmethod
    def from_df(cls, path, df:DataFrame, dep_var:str, valid_idx:Collection[int], procs:OptTabTfms=None,
                cat_names:OptStrList=None, cont_names:OptStrList=None, classes:Collection=None, 
                test_df=None, **kwargs)->DataBunch:
        "Create a `DataBunch` from `df` and `valid_idx` with `dep_var`."
        cat_names = ifnone(cat_names, []).copy()
        cont_names = ifnone(cont_names, [o for o in df if o not in cat_names and o not in dep_var])
        procs = listify(procs)
        src = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(valid_idx))
        src = src.label_from_df(cols=dep_var) if classes is None else src.label_from_df(cols=dep_var, classes=classes)
        if test_df is not None: src.add_test(TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names,
                                                                 processor = src.train.x.processor))
        return src.databunch(**kwargs)

class RNNDataBunch(DataBunch):
    "Create a plane `RNNDataBunch` suitable for training a multi-output regression model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, num_workers:int=0,
               device:torch.device=None, collate_fn:Callable=data_collate, dl_tfms:Optional[Collection[Callable]]=None,
               **kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
#         datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=bs, **kwargs) for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
