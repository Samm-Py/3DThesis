/****************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC                                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of 3dThesis. 3dThesis is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#pragma once
#include <string>
#include <vector>
#include <deque>
#include <list>
#include <climits>
#include <cfloat>

#include "oti_scalar.h"

using std::vector;
using std::string;
using std::deque;
using std::list;

#define PI 3.14159265358979323846

// Integration segment. Coordinates come from the (double) scan path, but the
// diffusion/weight/power fields are Real so they can carry the material- and
// beam-power derivatives.
struct int_seg {
	Real xb, yb, zb, phix, phiy, phiz, dtau, qmod;
	double wmod = 1.0; // lateral beam width factor of the source segment
};

// What is used to integrate
struct Nodes {
	size_t size = 0;
	// {xb,yb,zb} = coords
	// {phix,phiy,phiz} = diffusion
	// {dtau} = node weight
	// {expmod} = frontloads computation
	vector<Real> xb, yb, zb, phix, phiy, phiz, dtau, expmod;
};

// What is read in from the paths
struct path_seg{
	int smode;			//Segment mode (line melt | spot melt)
	double sx, sy, sz;	//Segment end coordinates
	double sqmod;		//Segment power modulation
	double sparam;		//Segment time parameter (speed | spot time )
	double swidth = 1.0;//Segment lateral beam width factor (multiplies beam ax/ay;
	                    //optional 7th path column, 1.0 when absent)
	double seg_time;	//Segment end time
};

// 3D coordinate
struct coord
{
	double x, y, z;
};

// Filename read into simulation
struct FileNames {
	string	name, dataDir, mode, material, beam, path;
	string	domain, output, settings;
	string	rank_suffix; // "" in serial; ".<rank>" under MPI so per-rank
	                     // snapshot slices don't clobber a shared filename
};

// Material constants. Thermophysical properties are Real so derivatives w.r.t.
// conductivity/density/specific-heat propagate (diffusivity a inherits them);
// the CET fit parameters stay double (post-processing only).
struct Material {
	Real kon; // Thermal Conductivty
	Real rho; // Density
	Real cps; // Specifc Heat
	Real T_liq; // Liquidus Temperature
	Real T_init; // Inital Temperature (Preheat/Ambient)
	Real a; // Thermal Diffusivity
	double cet_a, cet_n, cet_N0; // Parameters for CET
};

// Beam specific parameters. Power q is a Real design variable; beam shape and
// the adaptive nondimensional timestep stay double.
struct Beam {
	Real ax, ay;       // Lateral beam widths: Real so dT/d(sigma) propagates (DV_SIG)
	double az;         // Depth/absorption sigma: fixed material property, not a control
	double eff; // Absoprtion Efficiency
	Real q; // Beam Power
	double nond_dt; // Nondimensional Time
};

// Collection of simulation parameters
struct SimParams {
	// Mode
	string mode; // Snapshots, TemperatureHistory, Solidification
	
	// Snapshots
	vector<double> SnapshotTimes;

	// Solidification
	string tracking = "None";		// meltpool tracking mode
	double dt = 1e-5;			// timestep
	int out_freq = 1;		// output frequency
	double radiusCheck = 1.0;		// The radius to be checking perimeter points from (TODO::different behavior for >1 and 0)
	bool secondary = false;		// calculate secondary solidifiaction
};

// Domain paramters
struct Domain {
	// Domain numbers
	int xnum, ynum, znum, pnum;

	// Domain bounds 
	double xmin = DBL_MAX;
	double xmax = -DBL_MAX;
	double ymin = DBL_MAX;
	double ymax = -DBL_MAX;
	double zmin = DBL_MAX;
	double zmax = -DBL_MAX;

	// Domain resolution
	double xres, yres, zres;

	// Domain reflections
	bool use_BCs;
	int	BC_reflections;
	double BC_xmin, BC_xmax, BC_ymin, BC_ymax, BC_zmin;

	// Domain point file
	string pointsFile;
	vector<coord> points;
	bool customPoints;
};

// Fundamental settings
struct Settings {
	
	// Itegration settings
	double dtau_min, t_hist, p_hist, r_max;
	
	// Liquidus search via Newton method settings
	int max_iter;
	double dttest;
	
	// Scan Path Compression
	int compress;

	// Compute
	int thnum;
	bool use_PINT;

	// OTI seed segment: 0-based Path.txt DATA-row index (header excluded; the
	// solver's internal path prepends an origin spot, handled in seed_ctx) of
	// the segment whose controls (Q, sigma, v) carry the derivative seeds.
	// -1 (default) = the LAST segment, the pre-existing behavior. A
	// non-default value enables "seed segment j, observe at scan end" for
	// cross-segment influence Jacobians; incompatible with path compression
	// (combined far-history nodes cannot be zone-classified against a middle
	// seed), which Init enforces.
	int seed_seg;

	// MPI
	bool mpi_overlap;
};

// Some utility variables
struct Utility {
	double allScansEndTime = 0;		// Time when all scans are done
	double approxEndTime = 0;		// Approximate end time to simulation
	double maxWidthMod = 1.0;		// Max per-segment beam width factor over all
									// paths; sizes conservative radii (r_max, melt search)
	bool sol_finish = false;				// Has solidification finished
	bool do_sol = false;					// Do solidification calculation
};

// What variables to output
struct Output {
	bool x, y, z;
	bool T, T_hist;
	bool tSol, G, Gx, Gy, Gz, V, dTdt, eqFrac, depth, numMelt;
	bool RDF, mp_stats;
	bool H, Hx, Hy, Hz;
};

// Entire simulation data structure
struct Simdat{
	FileNames	files;
	Output		output;

	Material	material;

	vector<Beam> beams;
	vector<vector<path_seg>> paths;

	Domain		domain;
	SimParams	param;
	
	Settings	settings;

	Utility		util;

	// Should this rank print?
	bool print = true;
	// Is this running with MPI (actually using multiple ranks)?
	bool mpi = false;
};