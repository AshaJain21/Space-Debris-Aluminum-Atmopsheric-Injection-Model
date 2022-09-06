[lat_grid, lon_grid] = ndgrid(-90:0.1:90, -180:0.1:180);
[lon_size_row, lon_size_col]= size(lon_grid);
[lat_size_row, lat_size_col]= size(lat_grid);

reference_lons = reshape(lon_grid,[lon_size_row*lon_size_col,1]);
reference_lats= reshape(lat_grid,[lat_size_row*lat_size_col,1]);

test_lons = [-180:1:0];
test_lats = zeros(size(test_lons));
binareasize = 500000;
[t_lats, t_lons, t_counts] =hista(reference_lats, reference_lons, binareasize);

rastered_surface = hista_defineEqualAreaBins(reference_lats, reference_lons, binareasize)

[latbin, lonbin, count] = hista_fixedbinning(test_lats, test_lons, rastered_surface, reference_lons)

save('test', "latbin")
clear latbin
load('test.mat')