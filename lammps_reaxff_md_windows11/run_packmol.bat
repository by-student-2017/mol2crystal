
for %%1 in (input.inp) do (
  "packmol.exe" < %%1
  echo input file: %%1
)
