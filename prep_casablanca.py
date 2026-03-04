from datasets import concatenate_datasets, load_dataset, load_from_disk

casa_sets = [
    'horrid-qvc/CasablancaUAETest',
    'horrid-qvc/CasablancaMoroccoTest',
    'horrid-qvc/CasablancaJordanTest',
    'horrid-qvc/CasablancaAlgeriaTest',
    'horrid-qvc/CasablancaYemenTest',
    'horrid-qvc/CasablancaPalestineTest',
    'horrid-qvc/CasablancaMauritaniaTest',
    'horrid-qvc/CasablancaEgyptTest',
]

ds1 = load_dataset('horrid-qvc/CasablancaUAETest')['test']
ds2 = load_dataset('horrid-qvc/CasablancaMoroccoTest')['test']
ds3 = load_dataset('horrid-qvc/CasablancaJordanTest')['test']
ds4 = load_dataset('horrid-qvc/CasablancaAlgeriaTest')['test']
ds5 = load_dataset('horrid-qvc/CasablancaYemenTest')['test']
ds6 = load_dataset('horrid-qvc/CasablancaPalestineTest')['test']
ds7 = load_dataset('horrid-qvc/CasablancaMauritaniaTest')['test']
ds8 = load_dataset('horrid-qvc/CasablancaEgyptTest')['test']

casa_ds = concatenate_datasets([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8])
casa_ds.save_to_disk('horrid-qvc/CasablancaAllTest')


casa = load_from_disk('horrid-qvc/CasablancaAllTest')
print(casa)