===============
Getting started
===============

1. Clone the repo ``dep_tregex``

   .. code-block:: none

       $ git clone https://github.com/yandex/dep_tregex.git

2. Test the module.

   .. code-block:: none

       $ cd dep_tregex/
       $ python -m'dep_tregex'

   .. note::

       If you use python2.6 (or earlier version), you'll have to specify the main module manually:

       .. code-block:: none

           $ python -m'dep_tregex.__main__'

   .. code-block:: none

       usage: python -mdep_tregex [-h]
                                  {words,wc,nth,head,tail,shuf,grep,sed,html,gdb} ...

       positional arguments:
         {words,wc,nth,head,tail,shuf,grep,sed,html,gdb}
           words               extract words from tree
           wc                  count trees
           nth                 print only Nth tree
           head                print only first N trees
           tail                print only last N trees
           shuf                shuffle trees
           grep                filter trees by pattern
           sed                 apply tree scripts to trees
           html                view trees in browser
           gdb                 view step-by-step invocation

       optional arguments:
         -h, --help            show this help message and exit

3. Test a cooler feature.

   .. code-block:: none

       $ python -m'dep_tregex' html <en-ud-test.conllu

   A new browser window should open with some dependency trees in it.

   .. raw:: html

    <style type="text/css">
        svg { display: block; }
    </style>
    <style type="text/css">
      /* Generic */
      * { stroke: none; fill: none; transition: 0.1s ease-out, stroke-dasharray 0.02s; }
      text { font-family: sans-serif; text-anchor: middle; cursor: default; }
      .hid { opacity: 0.0; }
      rect.hid { fill: #000; stroke: #000; stroke-width: 10.00px; }
      path.hid { fill: none; stroke: #000; stroke-width: 10.00px; }

      /* Labels & arcs */
      .big { font-size: 12px; fill: #000; font-weight: bold; }
      .small { font-size: 10px; fill: #444; }
      .user-hl > .big   { fill: #048; }
      .user-hl > .small { fill: #039; }
      .role { font-size: 10px; fill: #444; font-style: italic; }
      .arc { stroke: black; stroke-width: 0.50px; }
      .arrow { fill: #000; }

      /* Label and arc highlight on hover */
      g:hover > text.big { fill: #f00; }
      g:hover > text.small { fill: #900; }
      g:hover > text.role { fill: #f00; }
      g:hover > path.arc { stroke: #f00; }
      g:hover > path.arrow { fill: #f00; }

      /* Arc highlight on label hover */
      g:hover + g > text.role { fill: #f00; }
      g:hover + g > path.arc { stroke: #f00; stroke-dasharray: 5,5; }
      g:hover + g > path.arrow { fill: #f00; }
    </style>
    <svg width="480" height="96" class="svg0">
      <style type="text/css">
        .svg0 .w4:hover ~ .w2 > text.big { fill: #c00; }
        .svg0 .w4:hover ~ .w2 > text.small { fill: #800; }
        .svg0 .w4:hover ~ .a2 > text.role { fill: #c00; }
        .svg0 .w4:hover ~ .a2 > path.arc { stroke: #c00; }
        .svg0 .w4:hover ~ .a2 > path.arrow { fill: #c00; }
        .svg0 .w1:hover ~ .w2 > text.big { fill: #888; }
        .svg0 .w1:hover ~ .w2 > text.small { fill: #666; }
        .svg0 .w1:hover ~ .a2 > text.role { fill: #888; }
        .svg0 .w1:hover ~ .a2 > path.arc { stroke: #888; }
        .svg0 .w1:hover ~ .a2 > path.arrow { fill: #888; }
        .svg0 .w4:hover ~ .w3 > text.big { fill: #c00; }
        .svg0 .w4:hover ~ .w3 > text.small { fill: #800; }
        .svg0 .w4:hover ~ .a3 > text.role { fill: #c00; }
        .svg0 .w4:hover ~ .a3 > path.arc { stroke: #c00; }
        .svg0 .w4:hover ~ .a3 > path.arrow { fill: #c00; }
        .svg0 .w1:hover ~ .w3 > text.big { fill: #888; }
        .svg0 .w1:hover ~ .w3 > text.small { fill: #666; }
        .svg0 .w1:hover ~ .a3 > text.role { fill: #888; }
        .svg0 .w1:hover ~ .a3 > path.arc { stroke: #888; }
        .svg0 .w1:hover ~ .a3 > path.arrow { fill: #888; }
        .svg0 .w1:hover ~ .w4 > text.big { fill: #c00; }
        .svg0 .w1:hover ~ .w4 > text.small { fill: #800; }
        .svg0 .w1:hover ~ .a4 > text.role { fill: #c00; }
        .svg0 .w1:hover ~ .a4 > path.arc { stroke: #c00; }
        .svg0 .w1:hover ~ .a4 > path.arrow { fill: #c00; }
        .svg0 .w6:hover ~ .w5 > text.big { fill: #c00; }
        .svg0 .w6:hover ~ .w5 > text.small { fill: #800; }
        .svg0 .w6:hover ~ .a5 > text.role { fill: #c00; }
        .svg0 .w6:hover ~ .a5 > path.arc { stroke: #c00; }
        .svg0 .w6:hover ~ .a5 > path.arrow { fill: #c00; }
        .svg0 .w4:hover ~ .w5 > text.big { fill: #888; }
        .svg0 .w4:hover ~ .w5 > text.small { fill: #666; }
        .svg0 .w4:hover ~ .a5 > text.role { fill: #888; }
        .svg0 .w4:hover ~ .a5 > path.arc { stroke: #888; }
        .svg0 .w4:hover ~ .a5 > path.arrow { fill: #888; }
        .svg0 .w1:hover ~ .w5 > text.big { fill: #888; }
        .svg0 .w1:hover ~ .w5 > text.small { fill: #666; }
        .svg0 .w1:hover ~ .a5 > text.role { fill: #888; }
        .svg0 .w1:hover ~ .a5 > path.arc { stroke: #888; }
        .svg0 .w1:hover ~ .a5 > path.arrow { fill: #888; }
        .svg0 .w4:hover ~ .w6 > text.big { fill: #c00; }
        .svg0 .w4:hover ~ .w6 > text.small { fill: #800; }
        .svg0 .w4:hover ~ .a6 > text.role { fill: #c00; }
        .svg0 .w4:hover ~ .a6 > path.arc { stroke: #c00; }
        .svg0 .w4:hover ~ .a6 > path.arrow { fill: #c00; }
        .svg0 .w1:hover ~ .w6 > text.big { fill: #888; }
        .svg0 .w1:hover ~ .w6 > text.small { fill: #666; }
        .svg0 .w1:hover ~ .a6 > text.role { fill: #888; }
        .svg0 .w1:hover ~ .a6 > path.arc { stroke: #888; }
        .svg0 .w1:hover ~ .a6 > path.arrow { fill: #888; }
        .svg0 .w4:hover ~ .w7 > text.big { fill: #c00; }
        .svg0 .w4:hover ~ .w7 > text.small { fill: #800; }
        .svg0 .w4:hover ~ .a7 > text.role { fill: #c00; }
        .svg0 .w4:hover ~ .a7 > path.arc { stroke: #c00; }
        .svg0 .w4:hover ~ .a7 > path.arrow { fill: #c00; }
        .svg0 .w1:hover ~ .w7 > text.big { fill: #888; }
        .svg0 .w1:hover ~ .w7 > text.small { fill: #666; }
        .svg0 .w1:hover ~ .a7 > text.role { fill: #888; }
        .svg0 .w1:hover ~ .a7 > path.arc { stroke: #888; }
        .svg0 .w1:hover ~ .a7 > path.arrow { fill: #888; }
      </style>
      <g class="w1">
        <rect x="12" y="72" width="48" height="12" class="hid" />
        <text x="36" y="84" class="big">What</text>
      </g>
      <g class="a1">
        <path d="M 36 72 L 36 24" class="arc" />
        <path d="M 36 72 L 36 24" class="arc hid" />
        <path d="M 36.00 73.50L 37.94 66.26L 36.00 67.12L 34.06 66.26Z" class="arrow"/>
        <text x="36" y="22" class="role">root</text>
      </g>
      <g class="w4">
        <rect x="192" y="72" width="84" height="12" class="hid" />
        <text x="234" y="84" class="big">Morphed</text>
      </g>
      <g class="a4">
        <path d="M 42.00 72.00A 72.00 72.00 0 0 1 104.35 36.00L 171.65 36.00A 72.00 72.00 0 0 1 234.00 72.00" class="arc" />
        <path d="M 42.00 72.00A 72.00 72.00 0 0 1 104.35 36.00L 171.65 36.00A 72.00 72.00 0 0 1 234.00 72.00" class="arc hid" />
        <path d="M 234.75 73.30L 232.81 66.05L 231.56 67.78L 229.45 68.00Z" class="arrow"/>
        <text x="138" y="34" class="role">advcl</text>
      </g>
      <g class="w2">
        <rect x="72" y="72" width="24" height="12" class="hid" />
        <text x="84" y="84" class="big">if</text>
      </g>
      <g class="a2">
        <path d="M 84.00 72.00A 48.00 48.00 0 0 1 125.57 48.00L 186.43 48.00A 48.00 48.00 0 0 1 228.00 72.00" class="arc" />
        <path d="M 84.00 72.00A 48.00 48.00 0 0 1 125.57 48.00L 186.43 48.00A 48.00 48.00 0 0 1 228.00 72.00" class="arc hid" />
        <path d="M 83.25 73.30L 88.55 68.00L 86.44 67.78L 85.19 66.05Z" class="arrow"/>
        <text x="156" y="46" class="role">mark</text>
      </g>
      <g class="w3">
        <rect x="108" y="72" width="72" height="12" class="hid" />
        <text x="144" y="84" class="big">Google</text>
      </g>
      <g class="a3">
        <path d="M 144.00 72.00A 24.00 24.00 0 0 1 164.78 60.00L 207.22 60.00A 24.00 24.00 0 0 1 228.00 72.00" class="arc" />
        <path d="M 144.00 72.00A 24.00 24.00 0 0 1 164.78 60.00L 207.22 60.00A 24.00 24.00 0 0 1 228.00 72.00" class="arc hid" />
        <path d="M 143.25 73.30L 148.55 68.00L 146.44 67.78L 145.19 66.05Z" class="arrow"/>
        <text x="186" y="58" class="role">nsubj</text>
      </g>
      <g class="w6">
        <rect x="348" y="72" width="96" height="12" class="hid" />
        <text x="396" y="84" class="big">GoogleOS</text>
      </g>
      <g class="a6">
        <path d="M 240.00 72.00A 48.00 48.00 0 0 1 281.57 48.00L 354.43 48.00A 48.00 48.00 0 0 1 396.00 72.00" class="arc" />
        <path d="M 240.00 72.00A 48.00 48.00 0 0 1 281.57 48.00L 354.43 48.00A 48.00 48.00 0 0 1 396.00 72.00" class="arc hid" />
        <path d="M 396.75 73.30L 394.81 66.05L 393.56 67.78L 391.45 68.00Z" class="arrow"/>
        <text x="318" y="46" class="role">nmod</text>
      </g>
      <g class="w7">
        <rect x="456" y="72" width="12" height="12" class="hid" />
        <text x="462" y="84" class="big">?</text>
      </g>
      <g class="a7">
        <path d="M 240.00 72.00A 72.00 72.00 0 0 1 302.35 36.00L 399.65 36.00A 72.00 72.00 0 0 1 462.00 72.00" class="arc" />
        <path d="M 240.00 72.00A 72.00 72.00 0 0 1 302.35 36.00L 399.65 36.00A 72.00 72.00 0 0 1 462.00 72.00" class="arc hid" />
        <path d="M 462.75 73.30L 460.81 66.05L 459.56 67.78L 457.45 68.00Z" class="arrow"/>
        <text x="351" y="34" class="role">punct</text>
      </g>
      <g class="w5">
        <rect x="288" y="72" width="48" height="12" class="hid" />
        <text x="312" y="84" class="big">Into</text>
      </g>
      <g class="a5">
        <path d="M 312.00 72.00A 24.00 24.00 0 0 1 332.78 60.00L 369.22 60.00A 24.00 24.00 0 0 1 390.00 72.00" class="arc" />
        <path d="M 312.00 72.00A 24.00 24.00 0 0 1 332.78 60.00L 369.22 60.00A 24.00 24.00 0 0 1 390.00 72.00" class="arc hid" />
        <path d="M 311.25 73.30L 316.55 68.00L 314.44 67.78L 313.19 66.05Z" class="arrow"/>
        <text x="351" y="58" class="role">case</text>
      </g>
    </svg>

4. You're all set (just don't leave the ``dep_tregex/`` folder, or just add it to your PYTHONPATH).
