%Equal Area Binning of Reentry Locations

function bins = equalAreaBinning(iteration)
    if (iteration == 1)
        binarea = 100000; %km^2
        [rastered_surface, reference_lons] = compute_fixed_bins(binarea)
    else
        load ('reference_bins.mat' ,'rastered_surface', 'reference_lons')
    end
    data = readtable(strcat('./MonteCarloFiles/',num2str(iteration),'/propogated_satellites_',num2str(iteration),'.csv'));
    [latbin,lonbin,count] = hista_fixedbinning(data.('PredictedReentryLatitude'),data.('PredictedReentryLongitude'),rastered_surface, reference_lons);
    bins = [latbin,lonbin,count];
    writematrix(bins, strcat('./MonteCarloFiles/',num2str(iteration),'/binned_positions_',num2str(iteration),'.csv'));
end

function [rastered_surface, reference_lons] = compute_fixed_bins(binarea)
    [lat_grid, lon_grid] = ndgrid(-90:0.1:90, -180:0.1:180);
    [lon_size_row, lon_size_col]= size(lon_grid);
    [lat_size_row, lat_size_col]= size(lat_grid);
    reference_lons = reshape(lon_grid,[lon_size_row*lon_size_col,1]);
    reference_lats= reshape(lat_grid,[lat_size_row*lat_size_col,1]);
    rastered_surface = hista_defineEqualAreaBins(reference_lats, reference_lons, binarea)
    save('reference_bins.mat', 'rastered_surface', 'reference_lons')
end