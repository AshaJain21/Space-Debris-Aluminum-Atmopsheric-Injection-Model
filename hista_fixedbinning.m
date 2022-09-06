function [latbin, lonbin, count] = hista_fixedbinning(lat, lon, rastered_surface, reference_long)

% Copyright 1996-2017 The MathWorks, Inc.
%Modified by Asha Jain

if ~isequal(size(lat),size(lon))
    error(message('map:validate:inconsistentSizes2','HISTA','LAT','LON'))
end

defaultSpheroid = referenceSphere('earth','km');
spheroid = defaultSpheroid;
angleUnit = 'degrees';

if map.geodesy.isDegree(angleUnit)
    validateattributes(lat,{'double','single'},{'real','finite','>=',-90,'<=',90})
else
    validateattributes(lat,{'double','single'},{'real','finite','>=',-pi,'<=',pi})
end
validateattributes(lon,{'double','single'},{'real','finite'})

% Work in degrees; ensure column vectors.
[lat, lon] = toDegrees(angleUnit, lat(:), lon(:));

% Define count the instances in defined bins
[latbin, lonbin, count] = eqabin(lat, lon, spheroid, reference_long, rastered_surface);

% Convert the mesh to the specified angle unit.
[latbin, lonbin] = fromDegrees(angleUnit, latbin, lonbin);

%-----------------------------------------------------------------------

function [latbin, lonbin, count] ...
    = eqabin(lat, lon, spheroid, reference_long, rastered_surface)
% Define equal area cells, count the number of (lat,lon) points per cells,
% and return the centers of non-empty cells ("bins") as vectors in latitude
% and longitude, along with the count per bin.

if isempty(lat)
    latbin = [];
    lonbin = [];
    count = [];
else
    [converter, longitudeOrigin] = setUpProjection(spheroid,reference_long);
    [x,y] = forwardProject(converter, longitudeOrigin, lat, lon);
    %cellwidth = cellWidthInRadians(binAreaInSquareKilometers, spheroid);
    %R = rasterReferenceInEqualAreaSystem(x, y, cellwidth);
    [xbin, ybin, count] = pointsPerCell(rastered_surface,x,y);
    [latbin, lonbin] = inverseProject(converter, longitudeOrigin, xbin, ybin);
end

%-----------------------------------------------------------------------

function [converter, longitudeOrigin] = setUpProjection(spheroid,lon)
% Set up an equal area cylindrical projection defined by an authalic
% latitude converter and a longitude origin in degrees.

converter = map.geodesy.AuthalicLatitudeConverter(spheroid);
[~, lonlim] = geoquadpt(zeros(size(lon)), lon);
longitudeOrigin = centerlon(lonlim);

%-----------------------------------------------------------------------

function [x,y] = forwardProject(converter, longitudeOrigin, lat, lon)
% Map latitude-longitude locations to equal area cylindrical coordinates.

x = deg2rad(wrapTo180(lon - longitudeOrigin));
y = sind(forward(converter, lat, 'degrees'));

%-----------------------------------------------------------------------

function [lat, lon] = inverseProject(converter, longitudeOrigin, x, y)
% Unproject equal area cylindrical coordinates to latitude-longitude.

lat = inverse(converter, asind(y), 'degrees');
lon = longitudeOrigin + rad2deg(x);

%-----------------------------------------------------------------------

% function width = cellWidthInRadians(binAreaInSquareKilometers, spheroid)
% 
% if isprop(spheroid,'LengthUnit') && ~isempty(spheroid.LengthUnit)
%     lengthUnit = spheroid.LengthUnit;
%     kilometersPerUnit = unitsratio('km', lengthUnit);
%     radiusInKilometers = kilometersPerUnit * spheroid.SemimajorAxis;
% else
%     % The length unit of the spheroid is unspecified; assume kilometers.
%     radiusInKilometers = spheroid.SemimajorAxis;
% end
% cellWidthInKilometers = sqrt(binAreaInSquareKilometers);
% width = cellWidthInKilometers/radiusInKilometers;

%-----------------------------------------------------------------------

function lon = centerlon(lonlim)
% Center of an interval in longitude
%
%   Accounts for wrapping.  Returns the longitude of the meridian halfway
%   from the western limit to the eastern limit, when traveling east.
%   All angles are in degrees.

lon = wrapTo180(lonlim(1) + wrapTo360(diff(lonlim))/2);

%-----------------------------------------------------------------------

function R = rasterReferenceInEqualAreaSystem(x, y, cellwidth)
% Define a the referencing object grid of cells into which to bin the input
% points, working in the equal area cylindrical system of x and y. This
% system runs from -pi to pi in x and -1 to 1 in y.

halfwidth = cellwidth/2;

xmin = max(min(x - halfwidth), -pi);
xmax = min(max(x + halfwidth),  pi);

ymin = max(min(y - halfwidth), -1);
ymax = min(max(y + halfwidth),  1);

nrows = ceil((ymax - ymin)/cellwidth);
ncols = ceil((xmax - xmin)/cellwidth);

R = maprasterref('RasterSize', [nrows ncols], ...
    'XWorldLimits',[xmin xmax],'YWorldLimits',[ymin ymax]);

%-----------------------------------------------------------------------

function [xbin, ybin, count] = pointsPerCell(R,x,y)
% Map each point to a cell and determine the number of points per non-empty
% cell. Return the center coordinates of each non-empty cell (aka "bin") in
% column vector xbin and ybin, with corresponding number of points in
% column vector count.

[rows, cols] = worldToDiscrete(R,x,y);
nrows = R.RasterSize(1);
[urows, ucols, count] = uniquesub(nrows, rows, cols);
[xbin, ybin] = intrinsicToWorld(R, ucols, urows);

%-----------------------------------------------------------------------

function [urows, ucols, count] = uniquesub(nrows, rows, cols)
% Given row and column indices into a nrows-by-M array, return vectors
% listing the row and column indices of each unique (row,col) pair, and the
% number of elements having those indices.

% Linear indices of row-column mapping results; k will typically contain
% many repeated values:
%   k = sub2ind([nrows ncols], rows, cols);
k = sort(rows + nrows * (cols - 1));

% Once the index k is sorted, determine the number of points per cell by
% finding the locations at which k jumps to a new value.
p = find(diff([0; k; Inf]) ~= 0);
count = diff(p);
p(end) = [];

% Row and column indices of non-empty cells:
%   [urows, ucols] = ind2sub([nrows ncols], unique(k));
index = k(p);
urows = 1 + rem(index-1, nrows);
ucols = 1 + (index - urows)/nrows;
