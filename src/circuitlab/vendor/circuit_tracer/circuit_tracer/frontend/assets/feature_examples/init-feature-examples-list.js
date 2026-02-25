window.initFeatureExamplesList = function({renderAll, visState, sel}){
  var sel = sel.select('.feature-example-list')
  renderAll.feature.fns.push(async () => {
    if (visState.feature.isDead) return sel.html(`Feature ${visState.feature.featureIndex} failed to load`)
    // // Put quantiles into cols to fill white space.
    // var cols = d3.range(Math.max(1, Math.floor(sel.node().offsetWidth/800))).map(d => [])
    // cols.forEach(col => col.y = 0)
    // visState.feature.examples_quantiles.forEach((d, i) => {
    //   var col = cols[d3.minIndex(cols, d => d.y)]
    //   col.push(d)
    //   col.y += d.examples.length + 2 // quantile header/whitespace is about 2× bigger than an example
    //   if (!i) col.y += 6
    // })
    // 
    var cols = [visState.feature.examples_quantiles]
    sel.html('')
      .appendMany('div.example-2-col', cols)
      .appendMany('div', d => d)
      .each(drawQuantile)    
  })

  function drawQuantile(quantile){
    var sel = d3.select(this)

    var quintileSel = sel.append('div.example-quantile')
      .append('span.quantile-title').text(quantile.quantile_name + ' ')

    sel.appendMany('div.ctx-container', quantile.examples).each(drawExample)
  }

  function maybeHexEscapedToBytes(token) { // -> number[]
    let ret = [];
    while (token.length) {
      if (/^\\x[0-9a-f]{2}/.exec(token)) {
        ret.push(parseInt(token.slice(2, 4), 16));
        token = token.slice(4);
      } else {
        ret.push(...new TextEncoder().encode(token[0]));
        token = token.slice(1);
      }
    }
    return ret;
  }
  function mergeHexEscapedMax(tokens, acts) {
    // -> {token: string, act: number, minIndex: int, maxIndex: int}[]
    let ret = [];
    let i = 0;
    while (i < tokens.length) {
      let maxAct = acts[i];
      let pushedMerge = false;
      if (/\\x[0-9a-f]{2}/.exec(tokens[i])) {
        let buf = maybeHexEscapedToBytes(tokens[i]);
        for (let j = i + 1; j < Math.min(i + 5, tokens.length); j++) {
          maxAct = Math.max(maxAct, acts[j]);
          buf.push(...maybeHexEscapedToBytes(tokens[j]));
          try {
            let text = new TextDecoder("utf-8", { fatal: true }).decode(
              new Uint8Array(buf),
            );
            ret.push({
              token: text,
              act: maxAct,
              minIndex: i,
              maxIndex: j,
            });
            i = j + 1;
            pushedMerge = true;
            break;
          } catch (e) {
            continue;
          }
        }
      }
      if (!pushedMerge) {
        ret.push({
          token: tokens[i],
          act: acts[i],
          minIndex: i,
          maxIndex: i,
        });
        i++;
      }
    }
    return ret;
  }

  function mergeConsecutiveSameActivations(tokens) {
    // Merge tokens with the same activation level
    const merged = [];
    let currentGroup = null;
    
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      
      if (!currentGroup) {
        // Start a new group
        currentGroup = {
          token: token.token,
          act: token.act,
          minIndex: token.minIndex,
          maxIndex: token.maxIndex
        };
      } else if (currentGroup.act === token.act) {
        // Same activation, merge with current group
        currentGroup.token += token.token;
        currentGroup.maxIndex = token.maxIndex;
      } else {
        // Different activation, push current group and start a new one
        merged.push(currentGroup);
        currentGroup = {
          token: token.token,
          act: token.act,
          minIndex: token.minIndex,
          maxIndex: token.maxIndex
        };
      }
    }
    
    // Don't forget to add the last group
    if (currentGroup) {
      merged.push(currentGroup);
    }
    
    return merged;
  }

  function drawExample(exp){
    var sel = d3.select(this).append('div')
      .st({opacity: exp.is_repeated_datapoint ? .4 : 1})
    var textSel = sel.append('div.text-wrapper')

    // First merge hex-escaped sequences
    var tokenData = mergeHexEscapedMax(exp.tokens, exp.tokens_acts_list);
    
    // Then merge consecutive tokens with the same activation
    tokenData = mergeConsecutiveSameActivations(tokenData);
    
    var tokenSel = textSel.appendMany('span.token', tokenData)
        .text(d => d.token)
        // .at({title: d => `${d.token} (${d.act})` })

    tokenSel
        .filter(d => d.act)
        .st({background: d => visState.feature.colorScale(d.act)})

    // We need to track if our merged token contains the train_token_ind
    var centerNode = tokenSel
        .filter(d => d.minIndex <= exp.train_token_ind && exp.train_token_ind <= d.maxIndex)
        .classed('train_token_ind', 1)
        .node()

    if (!centerNode) return
    
    const resizeObserver = new ResizeObserver(entries => {
      const selWidth = sel.node().offsetWidth;
      const nodeWidth = centerNode.offsetWidth;
      const nodeLeft = centerNode.offsetLeft;
      const leftOffset = (selWidth - nodeWidth)/2 - nodeLeft;
      textSel.translate([leftOffset, 0]);
    });
    resizeObserver.observe(sel.node());
  }
}

// window.initFeatureExample?.()
window.init?.()
