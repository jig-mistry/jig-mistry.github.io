
/* ---------------------------------------------------------------------------
   Background neural network.

   Neurons are grown with type-specific morphology (soma shape, dendritic field
   with spines, myelinated or unmyelinated axon ending in a terminal arbor),
   wired by proximity with excitatory and inhibitory weights, and simulated as
   leaky integrate-and-fire units.  A spike travels the axon at a finite speed
   and reaches each synapse after a conduction delay set by path length.

   The whole sheet drifts as one piece -- neurons do not swim around
   independently in tissue -- so positions are fixed relative to each other and
   a single transform moves, turns and breathes the entire web.
--------------------------------------------------------------------------- */
(function(){
  if (matchMedia('(pointer: coarse)').matches && innerWidth < 700) return;
  var REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Background quality. 'off' skips the whole thing; 'auto' starts full and
  // drops a level if frames get slow, which is the only honest way to cope
  // with an unknown machine and an unknown window size.
  var pref = 'auto';
  try { pref = localStorage.getItem('neuro.bg') || 'auto'; } catch (e) {}
  if (pref === 'off') return;
  var QUALITY = pref === 'lite' ? 1 : 0;        // 0 full, 1 lite, 2 minimal
  var LOCKED = pref !== 'auto';
  if (pref === 'auto'){
    try { QUALITY = Math.max(0, Math.min(2, +localStorage.getItem('neuro.bgauto') || 0)); }
    catch (e) {}
  }

  var TYPES = {
    pyramidal:  {p:'E', soma:'tri',  r:8.0, dend:5, dlen:80, axon:215, coll:3, w:0.34, mye:1, spine:1},
    stellate:   {p:'E', soma:'round',r:6.0, dend:6, dlen:50, axon:125, coll:3, w:0.30, mye:0, spine:1},
    granule:    {p:'E', soma:'round',r:4.5, dend:3, dlen:38, axon:235, coll:2, w:0.26, mye:1, spine:1},
    basket:     {p:'I', soma:'round',r:7.0, dend:5, dlen:56, axon:115, coll:4, w:0.52, mye:0, spine:0},
    chandelier: {p:'I', soma:'oval', r:6.4, dend:4, dlen:48, axon:100, coll:4, w:0.46, mye:0, spine:0},
    bipolar:    {p:'I', soma:'fusi', r:5.4, dend:2, dlen:62, axon:155, coll:2, w:0.38, mye:0, spine:0},
    unipolar:   {p:'E', soma:'round',r:5.6, dend:1, dlen:30, axon:195, coll:2, w:0.30, mye:1, spine:0},
    // Martinotti cells send their axon up towards layer 1; neurogliaform cells
    // make a tiny, extremely dense local plexus
    martinotti: {p:'I', soma:'oval', r:5.8, dend:4, dlen:54, axon:185, coll:3, w:0.44, mye:0, spine:0, up:1},
    neurogliaform:{p:'I', soma:'round',r:5.0, dend:9, dlen:26, axon:72, coll:3, w:0.34, mye:0, spine:0}
  };
  // Drawing each cell's type independently let the excitatory fraction wander
  // between 57% and 76% build to build. Cortex holds near 80:20, so the split
  // is fixed first and the type is drawn from the matching pool.
  var E_MIX = ['pyramidal','pyramidal','pyramidal','pyramidal','pyramidal',
               'stellate','stellate','granule','granule','unipolar'];
  var I_MIX = ['basket','basket','chandelier','bipolar','martinotti',
               'neurogliaform'];
  var E_FRACTION = 0.8;

  var rnd = Math.random;
  function jit(a, d){ return a + (rnd()-0.5)*d; }

  /* --- morphology, in soma-local coordinates --------------------------- */
  function branch(x, y, ang, len, depth, order, out, spread, taper){
    if (depth <= 0 || len < 5) return;
    var pts = [[x, y]], n = 3, cx = x, cy = y, a = ang;
    for (var i = 0; i < n; i++){
      a = jit(a, 0.38);
      cx += Math.cos(a) * len / n;
      cy += Math.sin(a) * len / n;
      pts.push([cx, cy]);
    }
    out.push({pts: pts, order: Math.min(3, order)});
    var kids = depth > 1 ? (rnd() < 0.82 ? 2 : 1) : (rnd() < 0.4 ? 1 : 0);
    for (var k = 0; k < kids; k++)
      branch(cx, cy, a + (k ? spread : -spread) * (0.5 + rnd()*0.7),
             len * (taper || 0.6) * (0.9 + rnd()*0.25), depth - 1, order + 1,
             out, spread, taper);
  }

  /* short ticks along the higher-order dendrites: spines read as texture and
     are what separates a spiny excitatory cell from a smooth interneuron */
  function spines(n){
    var out = [];
    if (!TYPES[n.type].spine) return out;
    for (var i = 0; i < n.dend.length; i++){
      var b = n.dend[i];
      if (b.order < 1) continue;
      for (var j = 1; j < b.pts.length; j++){
        var p0 = b.pts[j-1], p1 = b.pts[j];
        var seg = Math.hypot(p1[0]-p0[0], p1[1]-p0[1]);
        var count = Math.max(0, Math.round(seg / 9));
        for (var s = 0; s < count; s++){
          var u = (s + 0.5) / count;
          var px = p0[0] + (p1[0]-p0[0])*u, py = p0[1] + (p1[1]-p0[1])*u;
          var a = Math.atan2(p1[1]-p0[1], p1[0]-p0[0]) + (rnd() < 0.5 ? 1.57 : -1.57);
          var L = 2.2 + rnd()*1.6;
          out.push([px, py, px + Math.cos(a)*L, py + Math.sin(a)*L]);
        }
      }
    }
    return out;
  }

  function grow(n){
    var t = TYPES[n.type], d = [];
    if (n.type === 'pyramidal'){
      // apical trunk, then a tuft at the top, then the basal skirt
      var trunk = [[0,0]], y = 0;
      for (var s = 0; s < 4; s++){ y -= t.dlen*0.36; trunk.push([jit(0, 9), y]); }
      d.push({pts: trunk, order: 0});
      for (var u = 0; u < 3; u++)
        branch(trunk[4][0], trunk[4][1], jit(-Math.PI/2, 1.5), t.dlen*0.42, 2, 1, d, 0.7, 0.62);
      for (var i = 0; i < t.dend; i++)
        branch(0, 0, jit(Math.PI/2, 2.0), t.dlen*0.42, 2, 1, d, 0.75, 0.6);
    } else if (n.type === 'bipolar'){
      branch(0, 0, -Math.PI/2, t.dlen, 3, 0, d, 0.36, 0.6);
      branch(0, 0,  Math.PI/2, t.dlen*0.42, 2, 1, d, 0.3, 0.6);
    } else if (n.type === 'unipolar'){
      branch(0, 0, jit(-Math.PI/2, 0.5), t.dlen, 2, 0, d, 0.3, 0.6);
    } else {
      for (var j = 0; j < t.dend; j++)
        branch(0, 0, (j / t.dend) * Math.PI*2 + rnd()*0.6,
               t.dlen*(0.7+rnd()*0.5), 3, 0, d, 0.8, 0.58);
    }
    n.dend = d;
    n.spines = spines(n);
    n.dpts = [];
    for (var q = 0; q < d.length; q++)
      for (var p = 1; p < d[q].pts.length; p++) n.dpts.push(d[q].pts[p]);
  }

  /* --- glia and vasculature --------------------------------------------- */
  /* Astrocytes tile the tissue in non-overlapping domains and send an endfoot
     to the nearest capillary; oligodendrocytes reach out to the axons they
     myelinate; microglia are the only motile cell here -- they migrate slowly
     and extend and retract their processes constantly, surveying. */
  function growGlia(g){
    var procs = [], i, k, a, len, pts, cx, cy, ang;
    if (g.kind === 'astro'){
      var fibrous = rnd() < 0.3;          // white-matter type: long and straight
      var n = fibrous ? 6 + (rnd()*3|0) : 9 + (rnd()*5|0);
      for (i = 0; i < n; i++){
        a = (i/n)*6.283 + rnd()*0.5;
        len = fibrous ? 26 + rnd()*20 : 15 + rnd()*17;
        pts = [[0,0]]; cx = 0; cy = 0; ang = a;
        for (k = 0; k < 3; k++){
          ang = jit(ang, fibrous ? 0.16 : 0.55);
          cx += Math.cos(ang)*len/3; cy += Math.sin(ang)*len/3;
          pts.push([cx, cy]);
        }
        procs.push(pts);
        for (var t2 = 0; t2 < (fibrous ? 1 : 2); t2++){   // fine bushy tufts
          var b = [[cx, cy]], bx = cx, by = cy, ba = ang;
          for (k = 0; k < 2; k++){
            ba = jit(ba, 1.0);
            bx += Math.cos(ba)*len*0.3; by += Math.sin(ba)*len*0.3;
            b.push([bx, by]);
          }
          procs.push(b);
        }
      }
    } else if (g.kind === 'opc'){
      // NG2 glia: finer and even more ramified than microglia, and like them
      // they migrate -- the main dividing cell of the adult brain
      var np = 10 + (rnd()*5|0);
      for (i = 0; i < np; i++){
        a = (i/np)*6.283 + rnd()*0.5; len = 11 + rnd()*13;
        pts = [[0,0]]; cx = 0; cy = 0; ang = a;
        for (k = 0; k < 3; k++){
          ang = jit(ang, 0.9);
          cx += Math.cos(ang)*len/3; cy += Math.sin(ang)*len/3;
          pts.push([cx, cy]);
        }
        procs.push(pts);
      }
      g.phases = procs.map(function(){ return rnd()*6.283; });
    } else if (g.kind === 'peri'){
      // pericyte: a bump on the capillary with processes running along it
      var ta2 = g.tangent || 0;
      for (i = -1; i <= 1; i += 2){
        pts = [[0,0]]; cx = 0; cy = 0; ang = ta2 + (i > 0 ? 0 : Math.PI);
        for (k = 0; k < 3; k++){
          ang = jit(ang, 0.22);
          cx += Math.cos(ang)*9; cy += Math.sin(ang)*9;
          pts.push([cx, cy]);
        }
        procs.push(pts);
      }
    } else if (g.kind === 'micro'){
      var m = 7 + (rnd()*4|0);
      for (i = 0; i < m; i++){
        a = (i/m)*6.283 + rnd()*0.4; len = 13 + rnd()*12;
        pts = [[0,0]]; cx = 0; cy = 0; ang = a;
        for (k = 0; k < 3; k++){
          ang = jit(ang, 0.75);
          cx += Math.cos(ang)*len/3; cy += Math.sin(ang)*len/3;
          pts.push([cx, cy]);
        }
        procs.push(pts);
      }
      g.phases = procs.map(function(){ return rnd()*6.283; });
    } else {                                      // oligodendrocyte
      var o = 4 + (rnd()*3|0);
      for (i = 0; i < o; i++){
        a = (i/o)*6.283 + rnd()*0.3; len = 12 + rnd()*8;
        procs.push([[0,0], [Math.cos(a)*len*0.6, Math.sin(a)*len*0.6],
                    [Math.cos(a+0.2)*len, Math.sin(a+0.2)*len]]);
      }
    }
    g.procs = procs;
  }

  /* --- network ---------------------------------------------------------- */
  var W = 0, H = 0, DPR = 1, cells = [], spikes = [], flashes = [];
  var glia = [], vessels = [];
  var mx = -9999, my = -9999, running = true, last = 0, cv, ctx;
  var hotx = 0, hoty = 0, hotvx = 26, hotvy = 18, tglob = 0;
  // the sheet as a whole: offset, a little rotation, a little breathing
  var ox = 0, oy = 0, ovx = 0, ovy = 0, rot = 0, rotv = 0;
  var inhTone = 0;                  // how hard the interneurons are being worked
  var SPARK_TARGET = 70;            // spikes in flight the circuit settles at
  var FRAME_BUDGET = 9;             // ms per frame before the circuit calms down
  var frameCost = 0;                // smoothed cost of drawing a frame
  var scrollPos = 0, scrollPar = 0;

  function build(){
    var n = Math.max(10, Math.min(46, Math.round(W * H / (QUALITY ? 62000 : 34000))));
    cells = []; spikes = []; flashes = [];
    var nE = Math.round(n * E_FRACTION);
    for (var i = 0; i < n; i++){
      var pool = i < nE ? E_MIX : I_MIX;
      var type = pool[(rnd()*pool.length)|0], t = TYPES[type];
      var c = {type:type, pol:t.p, x: rnd()*W, y: rnd()*H,
               v: rnd()*0.4, out: [], fired: -1e9, glow: 0, phase: rnd()*6.283, exc: 1};
      grow(c);
      cells.push(c);
    }
    for (var a = 0; a < cells.length; a++){
      var src = cells[a], ta = TYPES[src.type], cand = [];
      for (var b = 0; b < cells.length; b++){
        if (a === b) continue;
        var dd = Math.hypot(cells[b].x-src.x, cells[b].y-src.y);
        // an ascending axon prefers targets above it
        if (ta.up && cells[b].y > src.y) dd *= 2.4;
        if (dd < ta.axon * 1.5) cand.push([dd, cells[b]]);
      }
      cand.sort(function(p,q){ return p[0]-q[0]; });
      var k = Math.min(ta.coll + 3, cand.length);
      for (var c2 = 0; c2 < k; c2++){
        var tgt = cand[c2][1];
        var idx = tgt.dpts.length ? (rnd()*tgt.dpts.length)|0 : -1;
        var ang = Math.atan2(tgt.y-src.y, tgt.x-src.x);
        var syn = {
          to: tgt, idx: idx,
          c1: [Math.cos(ang+0.55)*ta.axon*0.30, Math.sin(ang+0.55)*ta.axon*0.30],
          c2: [(rnd()-0.5)*44, (rnd()-0.5)*44],
          mye: ta.mye,
          // terminal arbor: a couple of short twigs and boutons at the contact
          twig: [[jit(0,2.4), jit(0,2.4)], [jit(0,2.4), jit(0,2.4)]],
          // Synaptic gain sits deliberately below self-sustaining. Above it the
          // network is bistable: the same settings measured 23 spikes/s on one
          // build and 0.7 on the next. Below it the drive sets the rate and it
          // holds within about +/-2 spikes/s across builds.
          w: (src.pol === 'E' ? ta.w * 0.9 : -ta.w * 1.5)
        };
        // cells never move relative to each other, so the axon's shape, its
        // length and its nodes of Ranvier can all be worked out once
        syn.path = axonPath(src, syn);
        syn.len = pathLen(syn.path);
        // eligibility only: whether it is actually sheathed depends on an
        // oligodendrocyte being near enough to wrap it (assigned below)
        syn.canMye = syn.mye;
        syn.mye = 0; syn.nodes = 0;
        src.out.push(syn);
      }
    }

    // capillaries: a couple of long faint curves the astrocytes attach to
    vessels = [];
    var nv = 1 + (rnd() < 0.6 ? 1 : 0);
    for (var v = 0; v < nv; v++){
      var vert = rnd() < 0.5;
      var pv = [];
      for (var q2 = 0; q2 <= 4; q2++){
        pv.push(vert ? [W*(0.15 + rnd()*0.7) + (q2-2)*18, H*q2/4]
                     : [W*q2/4, H*(0.15 + rnd()*0.7) + (q2-2)*18]);
      }
      vessels.push(pv);
    }

    glia = [];
    // pericytes sit on the capillaries, so place them along the vessel paths
    for (var pv2 = 0; pv2 < vessels.length; pv2++){
      var vp = vessels[pv2];
      var npc = 2 + (rnd()*3|0);
      for (var pc = 0; pc < npc; pc++){
        var fr = 0.12 + 0.76*((pc + rnd()*0.6) / Math.max(1, npc));
        var at1 = pointAt(vp, fr), at2 = pointAt(vp, Math.min(1, fr + 0.04));
        var pg = {kind:'peri', x: at1[0], y: at1[1], ca: 0, phase: rnd()*6.283,
                  vx: 0, vy: 0,
                  tangent: Math.atan2(at2[1]-at1[1], at2[0]-at1[0])};
        pg.hx = pg.x; pg.hy = pg.y;
        growGlia(pg);
        glia.push(pg);
      }
    }
    var ng = Math.round(cells.length * (QUALITY ? 0.6 : 1.1));
    for (var gi = 0; gi < ng; gi++){
      var roll = rnd();
      var kind = roll < 0.40 ? 'astro'
               : roll < 0.58 ? 'oligo'
               : roll < 0.82 ? 'micro' : 'opc';
      var g = {kind: kind, x: rnd()*W, y: rnd()*H, ca: 0, phase: rnd()*6.283,
               vx: 0, vy: 0};
      g.hx = g.x; g.hy = g.y;
      growGlia(g);
      if (kind === 'astro'){
        // endfoot to the nearest point on the nearest capillary
        var best = null, bd = 1e9;
        for (var vi = 0; vi < vessels.length; vi++){
          for (var pi = 0; pi < vessels[vi].length; pi++){
            var d2 = Math.hypot(vessels[vi][pi][0]-g.x, vessels[vi][pi][1]-g.y);
            if (d2 < bd){ bd = d2; best = vessels[vi][pi]; }
          }
        }
        if (best && bd < 210) g.foot = [best[0]-g.x, best[1]-g.y];
      }
      if (kind === 'oligo') g.wraps = [];
      glia.push(g);
    }

    // Myelin is not a property of an axon -- it is made by an oligodendrocyte,
    // a cell with its own soma and nucleus whose processes each wrap one
    // internode. So no axon is sheathed unless a cell is there to wrap it.
    var all = [];
    for (var ai = 0; ai < cells.length; ai++)
      for (var aj = 0; aj < cells[ai].out.length; aj++){
        var sy = cells[ai].out[aj];
        sy.mye = 0; sy.nodes = 0; sy.wrapped = null;
        if (sy.canMye && sy.len > 150) all.push(sy);
      }
    for (var oi = 0; oi < glia.length; oi++){
      var og = glia[oi];
      if (og.kind !== 'oligo') continue;
      var near = [];
      for (var ni = 0; ni < all.length; ni++){
        if (all[ni].mye) continue;
        var mid2 = pointAt(all[ni].path, 0.5);
        var md = Math.hypot(mid2[0]-og.x, mid2[1]-og.y);
        if (md < 205) near.push([md, all[ni], mid2]);
      }
      near.sort(function(a2,b2){ return a2[0]-b2[0]; });
      for (var wi = 0; wi < Math.min(7, near.length); wi++){
        var wsyn = near[wi][1], wmid = near[wi][2];
        wsyn.mye = 1;
        wsyn.nodes = Math.max(2, Math.round(wsyn.len / 22));
        wsyn.wrapped = og;
        og.wraps.push([wmid[0]-og.x, wmid[1]-og.y]);
      }
    }
  }

  function axonPath(c, s){
    var t = s.to;
    var ex = s.idx >= 0 ? t.x + t.dpts[s.idx][0] : t.x;
    var ey = s.idx >= 0 ? t.y + t.dpts[s.idx][1] : t.y;
    var p1 = [c.x + s.c1[0], c.y + s.c1[1]];
    var p2 = [(p1[0]+ex)/2 + s.c2[0], (p1[1]+ey)/2 + s.c2[1]];
    return [[c.x, c.y], p1, p2, [ex, ey]];
  }
  function pathLen(p){
    var L = 0;
    for (var i = 1; i < p.length; i++) L += Math.hypot(p[i][0]-p[i-1][0], p[i][1]-p[i-1][1]);
    return L;
  }
  function pointAt(p, frac){
    var L = pathLen(p), want = frac * L, acc = 0;
    for (var i = 1; i < p.length; i++){
      var d = Math.hypot(p[i][0]-p[i-1][0], p[i][1]-p[i-1][1]);
      if (acc + d >= want){
        var u = d ? (want - acc) / d : 0;
        return [p[i-1][0] + (p[i][0]-p[i-1][0])*u, p[i-1][1] + (p[i][1]-p[i-1][1])*u];
      }
      acc += d;
    }
    return p[p.length-1];
  }

  // createRadialGradient per soma per frame is one of canvas2d's slowest
  // calls; bake the glow once into a sprite and stamp it instead
  var sprites = {};
  function glowSprite(key, r, rgb, alpha){
    var id = key + '|' + (r|0) + '|' + rgb + '|' + alpha.toFixed(2);
    if (sprites[id]) return sprites[id];
    var c2 = document.createElement('canvas');
    var d2 = Math.max(2, (r*2)|0);
    c2.width = d2; c2.height = d2;
    var x2 = c2.getContext('2d');
    var g2 = x2.createRadialGradient(r, r, 0, r, r, r);
    g2.addColorStop(0, 'rgba(' + rgb + ',' + alpha + ')');
    g2.addColorStop(1, 'rgba(' + rgb + ',0)');
    x2.fillStyle = g2;
    x2.fillRect(0, 0, d2, d2);
    sprites[id] = c2;
    return c2;
  }

  function isDark(){
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return matchMedia('(prefers-color-scheme: dark)').matches;
  }

  /* --- simulation ------------------------------------------------------- */
  function step(dt, now){
    tglob += dt;
    // the sheet drifts and turns as one body
    ovx += (rnd()-0.5)*120*dt - ox*0.030*dt - ovx*0.50*dt;
    ovy += (rnd()-0.5)*120*dt - oy*0.030*dt - ovy*0.50*dt;
    ox += ovx*dt; oy += ovy*dt;
    rotv += (rnd()-0.5)*0.055*dt - rot*0.035*dt - rotv*0.5*dt;
    rot += rotv*dt;

    hotx += hotvx * dt; hoty += hotvy * dt;
    if (hotx < 0 || hotx > W) hotvx = -hotvx;
    if (hoty < 0 || hoty > H) hotvy = -hotvy;
    if (rnd() < dt * 0.25){
      var a = rnd()*6.283, sp = 18 + rnd()*34;
      hotvx = Math.cos(a)*sp; hotvy = Math.sin(a)*sp;
    }
    var wave = 0.55 + 0.45*Math.sin(tglob*0.21) * Math.cos(tglob*0.077 + 1.1);
    // Feedback inhibition, the way a real circuit does it: when traffic climbs,
    // the inhibitory cells are driven harder, they fire (the cold sparks), and
    // what they inhibit is the excitatory population. Nothing about the picture
    // changes -- only how hard the interneurons are working.
    // Two things recruit inhibition: too many spikes in flight, and frames
    // costing more than the budget. The second is the point -- when the page
    // gets heavy the circuit quietens itself instead of the wallpaper changing.
    var over = Math.max(spikes.length / SPARK_TARGET - 1,
                        (frameCost / FRAME_BUDGET - 1) * 1.6);
    inhTone += ((over > 0 ? over : -0.35) - inhTone) * Math.min(1, dt * 1.6);
    if (inhTone < 0) inhTone = 0;
    if (inhTone > 6) inhTone = 6;
    var busy = false;

    // the cursor lives in screen space; the sheet has moved under it
    var cs = Math.cos(-rot), sn = Math.sin(-rot);
    var rx = mx - (W/2 + ox), ry = my - (H/2 + oy);
    var cmx = rx*cs - ry*sn + W/2, cmy = rx*sn + ry*cs + H/2;

    for (var i = 0; i < cells.length; i++){
      var c = cells[i];
      c.v += (0 - c.v) * Math.min(1, dt * 1.15);
      if (c.pol === 'I') c.v += dt * 1.2 * inhTone;      // recruit interneurons
      else c.v -= dt * 0.5 * inhTone;                    // and they damp the rest
      // tonic inhibition rising in the wave's trough: modulating only the
      // spontaneous drive cannot quiet the network, since cascades sustain
      // themselves, so the down state needs a real hyperpolarising tone
      c.exc = 0.55 + 0.45*Math.sin(tglob*0.13 + c.phase);
      c.v -= dt * 3.0 * (1 - wave) * c.exc;
      var hd = Math.hypot(c.x - hotx, c.y - hoty);
      var focus = hd < 320 ? (1 - hd/320) : 0;
      if (!busy)
        c.v += (rnd() < dt * (1.4 + 2.24*wave*c.exc + 2.0*focus) ? 0.30 + rnd()*0.22 : 0);
      var d = Math.hypot(c.x - cmx, c.y - cmy);
      if (d < 190) c.v += (1 - d/190) * dt * 2.4;
      c.glow *= Math.pow(0.0016, dt);
      if (c.v >= 1 && now - c.fired > 0.35){
        c.fired = now; c.v = -0.25; c.glow = 1;
        // astrocytes answer nearby activity with a slow calcium transient,
        // seconds long rather than milliseconds
        for (var gg = 0; gg < glia.length; gg++){
          var ga = glia[gg];
          if (ga.kind !== 'astro') continue;
          var gd = Math.hypot(ga.x - c.x, ga.y - c.y);
          if (gd < 150) ga.ca = Math.min(1, ga.ca + 0.42*(1 - gd/150));
        }
        for (var k = 0; k < c.out.length; k++){
          var s = c.out[k];
          spikes.push({c: c, s: s, t: 0, dur: Math.max(0.14, s.len / 300)});
        }
      }
    }
    for (var gi = 0; gi < glia.length; gi++){
      var g = glia[gi];
      g.ca *= Math.pow(0.5, dt/2.6);                 // slow calcium decay
      if (g.kind === 'micro' || g.kind === 'opc'){
        // the motile cells: microglia survey briskly, NG2 glia wander slower.
        // Astrocytes, oligodendrocytes and pericytes stay put, as they do.
        var sp2 = g.kind === 'micro' ? 95 : 52;
        g.vx += (rnd()-0.5)*sp2*dt - (g.x-g.hx)*0.05*dt - g.vx*0.7*dt;
        g.vy += (rnd()-0.5)*sp2*dt - (g.y-g.hy)*0.05*dt - g.vy*0.7*dt;
        g.x += g.vx*dt; g.y += g.vy*dt;
      }
    }
    for (var q = spikes.length - 1; q >= 0; q--){
      var sp = spikes[q];
      sp.t += dt;
      if (sp.t >= sp.dur){
        sp.s.to.v += sp.s.w;
        var e = sp.s.path[3];
        flashes.push({x:e[0], y:e[1], t:0, exc: sp.s.w > 0});
        spikes.splice(q, 1);
      }
    }
    for (var f = flashes.length - 1; f >= 0; f--){
      flashes[f].t += dt;
      if (flashes[f].t > 0.45) flashes.splice(f, 1);
    }
  }

  /* --- drawing ---------------------------------------------------------- */
  function curve(p){
    ctx.moveTo(p[0][0], p[0][1]);
    for (var i = 1; i < p.length - 1; i++)
      ctx.quadraticCurveTo(p[i][0], p[i][1],
                           (p[i][0]+p[i+1][0])/2, (p[i][1]+p[i+1][1])/2);
    ctx.lineTo(p[p.length-1][0], p[p.length-1][1]);
  }

  function paint(){
    var dark = isDark();
    ctx.setTransform(DPR,0,0,DPR,0,0);
    ctx.clearRect(0,0,W,H);
    ctx.save();
    // one transform for the whole sheet
    var breathe = 1 + 0.006*Math.sin(tglob*0.09);
    // scrolling drags the sheet along behind the page. Bounded and smooth, so
    // a very long page cannot push the network off screen entirely.
    scrollPar += (110*Math.sin(scrollPos/700) - scrollPar) * 0.12;
    ctx.translate(W/2 + ox, H/2 + oy - scrollPar);
    ctx.rotate(rot);
    ctx.scale(breathe, breathe);
    ctx.translate(-W/2, -H/2);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    var i, j, c;

    // capillaries, furthest back
    ctx.beginPath();
    for (i = 0; i < vessels.length; i++) curve(vessels[i]);
    ctx.lineWidth = 7;
    ctx.strokeStyle = dark ? 'rgba(190,120,150,0.055)' : 'rgba(150,60,90,0.045)';
    ctx.stroke();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = dark ? 'rgba(230,150,175,0.05)' : 'rgba(140,50,80,0.05)';
    ctx.stroke();

    // glial processes: astrocytes bushy, microglia ramified and restless,
    // oligodendrocytes reaching to the internodes they wrap
    for (var gk = 0; gk < 5; gk++){
      var kind = ['astro','oligo','opc','peri','micro'][gk];
      ctx.beginPath();
      for (i = 0; i < glia.length; i++){
        var g = glia[i];
        if (g.kind !== kind) continue;
        for (j = 0; j < g.procs.length; j++){
          var pr = g.procs[j], wpts = [];
          // microglial processes extend and retract as they survey
          var f = (kind === 'micro' || kind === 'opc')
            ? (kind === 'micro' ? 0.62 + 0.38*Math.sin(tglob*0.9 + (g.phases ? g.phases[j] : 0))
                                : 0.74 + 0.26*Math.sin(tglob*0.5 + (g.phases ? g.phases[j] : 0)))
            : 1;
          for (var pp = 0; pp < pr.length; pp++)
            wpts.push([g.x + pr[pp][0]*f, g.y + pr[pp][1]*f]);
          curve(wpts);
        }
        if (g.foot){                       // astrocytic endfoot on a capillary
          ctx.moveTo(g.x, g.y);
          ctx.lineTo(g.x + g.foot[0], g.y + g.foot[1]);
        }
        if (g.wraps) for (j = 0; j < g.wraps.length; j++){
          // one process per internode it myelinates, bowed slightly so it
          // reads as a limb rather than a spoke
          var wx = g.x + g.wraps[j][0], wy = g.y + g.wraps[j][1];
          curve([[g.x, g.y],
                 [(g.x+wx)/2 + (wy-g.y)*0.12, (g.y+wy)/2 - (wx-g.x)*0.12],
                 [wx, wy]]);
        }
      }
      ctx.lineWidth = kind === 'oligo' ? 0.95 : (kind === 'peri' ? 1.5 : 0.7);
      ctx.strokeStyle =
          kind === 'astro' ? (dark ? 'rgba(150,205,210,0.135)' : 'rgba(40,85,90,0.115)')
        : kind === 'micro' ? (dark ? 'rgba(200,190,160,0.115)' : 'rgba(70,62,40,0.105)')
        : kind === 'opc'   ? (dark ? 'rgba(175,205,175,0.11)'  : 'rgba(48,78,48,0.10)')
        : kind === 'peri'  ? (dark ? 'rgba(225,160,175,0.16)'  : 'rgba(120,50,70,0.15)')
        :                    (dark ? 'rgba(205,170,215,0.125)' : 'rgba(80,50,95,0.11)');
      ctx.stroke();
    }

    // glial somata, with the astrocytes' calcium showing
    for (i = 0; i < glia.length; i++){
      var g2 = glia[i], ca = g2.ca;
      var gr = g2.kind === 'astro' ? 4.4
             : g2.kind === 'oligo' ? 3.6
             : g2.kind === 'opc' ? 2.9
             : g2.kind === 'peri' ? 3.4 : 3.2;
      ctx.beginPath();
      if (g2.kind === 'peri')   // elongated along the vessel it sits on
        ctx.ellipse(g2.x, g2.y, gr*1.7, gr*0.72, g2.tangent || 0, 0, 6.283);
      else
        ctx.arc(g2.x, g2.y, gr, 0, 6.283);
      ctx.fillStyle = g2.kind === 'astro'
        ? (dark ? 'rgba(' + (150+70*ca|0) + ',' + (215+30*ca|0) + ',215,' + (0.20+0.45*ca) + ')'
                : 'rgba(' + (40-20*ca|0) + ',' + (95-25*ca|0) + ',100,' + (0.22+0.4*ca) + ')')
        : g2.kind === 'oligo'
          ? (dark ? 'rgba(205,170,215,0.26)' : 'rgba(80,50,95,0.26)')
        : g2.kind === 'opc'
          ? (dark ? 'rgba(180,210,180,0.22)' : 'rgba(48,78,48,0.22)')
        : g2.kind === 'peri'
          ? (dark ? 'rgba(228,165,180,0.30)' : 'rgba(120,50,70,0.28)')
          : (dark ? 'rgba(205,195,165,0.24)' : 'rgba(70,62,40,0.24)');
      ctx.fill();
      // every glial cell is a cell: soma and nucleus, not an abstract dot
      ctx.beginPath(); ctx.arc(g2.x, g2.y, gr*0.42, 0, 6.283);
      ctx.fillStyle = dark ? 'rgba(30,26,22,0.42)' : 'rgba(252,248,242,0.42)';
      ctx.fill();
      if (ca > 0.08 && QUALITY < 2){        // slow calcium wave, seconds long
        var CR = 26 + 34*ca;
        var cs2 = glowSprite('a' + (dark ? 'd' : 'l'), 64,
                             dark ? '140,225,225' : '30,110,115',
                             dark ? 0.13 : 0.09);
        ctx.globalAlpha = ca;
        ctx.drawImage(cs2, g2.x - CR, g2.y - CR, CR*2, CR*2);
        ctx.globalAlpha = 1;
      }
    }

    var widths = [2.3, 1.5, 0.95, 0.6], alpha = [0.27, 0.21, 0.16, 0.12];
    for (var order = 0; order < 4; order++){
      ctx.beginPath();
      for (i = 0; i < cells.length; i++){
        c = cells[i];
        for (j = 0; j < c.dend.length; j++){
          if (c.dend[j].order !== order) continue;
          var pts = c.dend[j].pts, world = [];
          for (var p = 0; p < pts.length; p++) world.push([c.x+pts[p][0], c.y+pts[p][1]]);
          curve(world);
        }
      }
      ctx.lineWidth = widths[order];
      ctx.strokeStyle = dark ? 'rgba(255,166,128,' + alpha[order] + ')'
                             : 'rgba(70,32,16,' + alpha[order] + ')';
      ctx.stroke();
    }

    // dendritic spines: thousands of tiny sub-paths, first thing to go
    if (QUALITY < 1) {
    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.spines.length; j++){
        var sp2 = c.spines[j];
        ctx.moveTo(c.x + sp2[0], c.y + sp2[1]);
        ctx.lineTo(c.x + sp2[2], c.y + sp2[3]);
      }
    }
    ctx.lineWidth = 0.75;
    ctx.strokeStyle = dark ? 'rgba(255,170,132,0.13)' : 'rgba(70,32,16,0.13)';
    ctx.stroke();
    }

    // axon core: every axon, thin and continuous
    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.out.length; j++) curve(c.out[j].path);
    }
    ctx.lineWidth = 0.85;
    ctx.strokeStyle = dark ? 'rgba(255,126,86,0.17)' : 'rgba(96,44,20,0.18)';
    ctx.stroke();

    // myelin: internodal sheaths laid over the core, with the gaps between
    // them left bare -- those gaps are the nodes of Ranvier the spikes jump
    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.out.length; j++) if (c.out[j].mye) curve(c.out[j].path);
    }
    ctx.setLineDash([15, 6]);
    ctx.lineCap = 'butt';
    ctx.lineWidth = 3.4;
    ctx.strokeStyle = dark ? 'rgba(255,196,164,0.13)' : 'rgba(120,70,40,0.12)';
    ctx.stroke();
    ctx.lineWidth = 1.9;
    ctx.strokeStyle = dark ? 'rgba(255,214,190,0.10)' : 'rgba(150,100,66,0.10)';
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineCap = 'round';

    // axon hillock / initial segment: thicker where the axon leaves the soma
    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.out.length; j++){
        var hp = c.out[j].path;
        var hx2 = hp[0][0] + (hp[1][0]-hp[0][0])*0.09;
        var hy2 = hp[0][1] + (hp[1][1]-hp[0][1])*0.09;
        ctx.moveTo(hp[0][0], hp[0][1]); ctx.lineTo(hx2, hy2);
      }
    }
    ctx.lineWidth = 2.6;
    ctx.strokeStyle = dark ? 'rgba(255,150,110,0.20)' : 'rgba(86,40,18,0.20)';
    ctx.stroke();

    // terminal arbor: twigs and boutons at each contact, plus varicosities
    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.out.length; j++){
        var pa = c.out[j].path, e = pa[3], tw = c.out[j].twig;
        ctx.moveTo(e[0], e[1]); ctx.lineTo(e[0]+tw[0][0]*2.2, e[1]+tw[0][1]*2.2);
        ctx.moveTo(e[0], e[1]); ctx.lineTo(e[0]+tw[1][0]*2.2, e[1]+tw[1][1]*2.2);
      }
    }
    ctx.lineWidth = 0.7;
    ctx.strokeStyle = dark ? 'rgba(255,140,100,0.18)' : 'rgba(90,40,18,0.18)';
    ctx.stroke();

    ctx.beginPath();
    for (i = 0; i < cells.length; i++){
      c = cells[i];
      for (j = 0; j < c.out.length; j++){
        var pb = c.out[j].path;
        var e2 = pb[3];
        ctx.moveTo(e2[0]+2.0, e2[1]); ctx.arc(e2[0], e2[1], 2.0, 0, 6.283);
        var v1 = pointAt(pb, 0.45), v2 = pointAt(pb, 0.75);   // varicosities
        ctx.moveTo(v1[0]+1.1, v1[1]); ctx.arc(v1[0], v1[1], 1.1, 0, 6.283);
        ctx.moveTo(v2[0]+1.1, v2[1]); ctx.arc(v2[0], v2[1], 1.1, 0, 6.283);
      }
    }
    ctx.fillStyle = dark ? 'rgba(255,150,110,0.22)' : 'rgba(80,36,18,0.20)';
    ctx.fill();

    for (i = 0; i < cells.length; i++){
      c = cells[i];
      var t = TYPES[c.type], g = Math.min(1, Math.max(0, c.glow));
      var lit = Math.max(0, Math.min(1, c.v)) * 0.5 + g;
      var rg = ctx.createRadialGradient(c.x - t.r*0.3, c.y - t.r*0.3, t.r*0.15,
                                        c.x, c.y, t.r*1.25);
      if (dark){
        rg.addColorStop(0, 'rgba(255,' + (170+60*lit|0) + ',' + (130+70*lit|0) + ',' +
                            (0.32 + 0.55*lit) + ')');
        rg.addColorStop(1, 'rgba(255,120,80,' + (0.12 + 0.30*lit) + ')');
      } else {
        rg.addColorStop(0, 'rgba(' + (120-60*lit|0) + ',40,18,' + (0.32 + 0.5*lit) + ')');
        rg.addColorStop(1, 'rgba(90,40,20,' + (0.12 + 0.28*lit) + ')');
      }
      ctx.fillStyle = rg;
      somaPath(c, 1 + g*0.28);
      ctx.fill();
      ctx.fillStyle = dark ? 'rgba(90,40,25,0.30)' : 'rgba(255,250,244,0.35)';
      ctx.beginPath(); ctx.arc(c.x, c.y, t.r*0.34, 0, 6.283); ctx.fill();
      ctx.fillStyle = dark ? 'rgba(255,206,180,0.35)' : 'rgba(60,26,12,0.30)';
      ctx.beginPath(); ctx.arc(c.x + t.r*0.08, c.y - t.r*0.05, t.r*0.13, 0, 6.283);
      ctx.fill();
      if (g > 0.05 && QUALITY < 2){
        var R = t.r*7*g + 8;
        var sp3 = glowSprite('n' + (dark ? 'd' : 'l'), 64,
                             dark ? '255,150,105' : '190,60,25',
                             dark ? 0.30 : 0.16);
        ctx.globalAlpha = g;
        ctx.drawImage(sp3, c.x - R, c.y - R, R*2, R*2);
        ctx.globalAlpha = 1;
      }
    }

    for (i = 0; i < spikes.length; i++){
      var s3 = spikes[i], pc = s3.s.path, frac = s3.t / s3.dur;
      var L = s3.s.len;
      // a myelinated axon conducts saltatorily: the impulse jumps node to
      // node rather than gliding, so snap it to the nearest node of Ranvier
      var fr = Math.min(1, frac);
      if (s3.s.nodes) fr = Math.floor(fr * s3.s.nodes) / s3.s.nodes;
      var b2 = pointAt(pc, fr);
      var a3 = pointAt(pc, Math.max(0, fr - (s3.s.nodes ? 1/s3.s.nodes : 24/Math.max(L,1))));
      var exc = s3.s.w > 0;
      ctx.strokeStyle = dark
        ? (exc ? 'rgba(255,178,140,0.92)' : 'rgba(150,196,255,0.82)')
        : (exc ? 'rgba(190,50,18,0.78)'  : 'rgba(30,70,150,0.68)');
      ctx.lineWidth = 2.1;
      ctx.beginPath(); ctx.moveTo(a3[0], a3[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
      ctx.fillStyle = dark ? 'rgba(255,224,200,0.95)' : 'rgba(150,35,12,0.85)';
      ctx.beginPath(); ctx.arc(b2[0], b2[1], 2.1, 0, 6.283); ctx.fill();
    }

    for (i = 0; i < flashes.length; i++){
      var fl = flashes[i], u = 1 - fl.t/0.45;
      ctx.fillStyle = dark
        ? (fl.exc ? 'rgba(255,190,150,'+(0.5*u)+')' : 'rgba(150,200,255,'+(0.45*u)+')')
        : (fl.exc ? 'rgba(190,60,25,'+(0.35*u)+')'  : 'rgba(30,70,150,'+(0.3*u)+')');
      ctx.beginPath(); ctx.arc(fl.x, fl.y, 2 + (1-u)*7, 0, 6.283); ctx.fill();
    }
    ctx.restore();
  }

  function somaPath(c, k){
    var t = TYPES[c.type], r = t.r * k;
    ctx.beginPath();
    if (t.soma === 'tri'){
      ctx.moveTo(c.x, c.y - r*1.3);
      ctx.quadraticCurveTo(c.x + r*1.1, c.y + r*0.9, c.x, c.y + r*0.75);
      ctx.quadraticCurveTo(c.x - r*1.1, c.y + r*0.9, c.x, c.y - r*1.3);
    } else if (t.soma === 'fusi'){
      ctx.ellipse(c.x, c.y, r*0.62, r*1.5, 0, 0, 6.283);
    } else if (t.soma === 'oval'){
      ctx.ellipse(c.x, c.y, r*1.2, r*0.82, 0, 0, 6.283);
    } else {
      ctx.arc(c.x, c.y, r, 0, 6.283);
    }
  }

  /* --- loop and lifecycle ----------------------------------------------- */
  var frames = 0, sumMs = 0, slow = 0, reported = false, lastPaint = 0;
  /* The handle matters. Clearing `running` stops the *next* frame but leaves
     one already queued, and showing the tab started a second loop beside it,
     so every hide/show cycle permanently added another full-canvas repaint. */
  var rafId = 0;
  function frame(ts){
    if (!running){ rafId = 0; return; }
    rafId = requestAnimationFrame(frame);
    var now = ts / 1000;
    var minGap = QUALITY ? 1/31 : 0;      // lite runs at 30fps
    if (minGap && lastPaint && now - lastPaint < minGap) return;
    var dt = Math.min(0.05, lastPaint ? now - lastPaint : 0.016);
    lastPaint = now;

    var t0 = performance.now();
    if (!REDUCED) step(dt, now);
    paint();
    var cost = performance.now() - t0;
    frames++; sumMs += cost;
    frameCost += (cost - frameCost) * 0.10;    // feeds the inhibitory controller
    if (cost > 12) slow++;

    // The picture never changes underneath you. If this machine cannot afford
    // the current level, that is recorded and takes effect the next time the
    // tab is loaded -- never mid-session.
    if (!LOCKED && frames === 600){
      var want = slow > 240 ? Math.min(2, QUALITY + 1) : QUALITY;
      try { localStorage.setItem('neuro.bgauto', String(want)); } catch (e) {}
      if (want !== QUALITY)
        log('background will start at level ' + want + ' next reload (' +
            (sumMs/frames).toFixed(1) + ' ms/frame here)');
    }
    if (!reported && frames === 240){
      reported = true;
      log('background: level ' + QUALITY + ', ' + (sumMs/frames).toFixed(2) +
          ' ms/frame over ' + frames + ' frames, ' + cells.length + ' neurons, ' +
          glia.length + ' glia, dpr ' + DPR);
    }
  }

  function log(msg){
    // Frame timings. These used to go to the server this came from;
    // here they go nowhere. Set window.NEURO_LOG = console.log to
    // watch the quality level settle on a machine you care about.
    try { if (window.NEURO_LOG) window.NEURO_LOG('neuro: ' + msg); } catch (e) {}
  }

  function size(){
    // the background never needs retina: at DPR 2 every fill costs 4x
    DPR = QUALITY ? 1 : Math.min(1.35, devicePixelRatio || 1);
    W = innerWidth; H = innerHeight;
    cv.width = Math.round(W*DPR); cv.height = Math.round(H*DPR);
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
  }

  function boot(){
    cv = document.createElement('canvas'); cv.className = 'neuro';
    document.body.insertBefore(cv, document.body.firstChild);
    ctx = cv.getContext('2d');
    size();
    hotx = W*0.5; hoty = H*0.4;
    build();
    requestAnimationFrame(frame);
    var rt;
    addEventListener('resize', function(){
      clearTimeout(rt);
      rt = setTimeout(function(){
        // Only re-grow the network when the window really changed shape.
        // Pressing Alt reveals the menu bar, which nudges the viewport by a few
        // pixels -- rebuilding on that swapped the whole field for a new one.
        var grew = Math.abs(innerWidth - W) > W*0.2 ||
                   Math.abs(innerHeight - H) > H*0.2;
        size();
        if (grew) build();
      }, 300);
    });
    addEventListener('pointermove', function(ev){ mx = ev.clientX; my = ev.clientY; },
                     {passive:true});
    addEventListener('scroll', function(){
      scrollPos = window.scrollY || document.documentElement.scrollTop || 0;
    }, {passive:true});
    document.addEventListener('visibilitychange', function(){
      if (document.hidden){
        running = false;
        if (rafId){ cancelAnimationFrame(rafId); rafId = 0; }
      } else if (!running && !rafId){
        running = true; last = 0; rafId = requestAnimationFrame(frame);
      }
    });
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
