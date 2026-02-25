window.util = (function () {
  var params = (function(){
    var rv = {}

    rv.get = key => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      var str = searchParams.get(key)
      return str && decodeURIComponent(str)
    }

    rv.getAll = () => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      var values = {}
      for (const [key, value] of searchParams.entries()) {
        values[key] = decodeURIComponent(value)
      }
      return values
    }

    rv.set = (key, value) => {
      var url = new URL(window.location)
      var searchParams = new URLSearchParams(url.search)

      if (value === null) {
        searchParams.delete(key)
      } else {
        searchParams.set(key, value)
      }

      url.search = searchParams.toString()
      history.replaceState(null, '', url)
    }

    return rv
  })()
  
  async function getFile(path, useCache = true, fileType = null, range = null) {
    // Cache storage 
    var __datacache = window.__datacache = window.__datacache || {}

    if (path.startsWith('./features/')) {
      path = path.replace('./features/', 'https://d1fk9w8oratjix.cloudfront.net/features/')
    }

    if (!window.isLocalServing){
      if (window.location.hostname === 'localhost' && path.startsWith('./data/')) {
        path = path.replace('./data/', 'https://d1fk9w8oratjix.cloudfront.net/data/')
      }

      if (window.location.hostname === 'localhost' && path.startsWith('./graph_data/')) {
        path = path.replace('./graph_data/', 'https://d1fk9w8oratjix.cloudfront.net/graph_data/')
      }
    }
    
    // Return cached result if available 
    var cacheKey = path + (range ? `-${range}` : '') + (fileType ? `-${fileType}` : '')
    if (!useCache || !__datacache[cacheKey]) __datacache[cacheKey] = __fetch()
    return __datacache[cacheKey]

    async function __fetch() {
      var cacheOption = useCache ? 'force-cache' : 'no-cache'
      var headers = range ? {'Range': range} : {}
      var res = await fetch(path, {cache: cacheOption, headers})

      if (res.status == 500) {
        var resText = await res.text()
        console.log(resText, res) 
        throw '500 error'
      }

      var type = fileType || path.replaceAll('..', '').split('.').at(-1)
      if (type == 'csv') {
        return d3.csvParse(await res.text())
      } else if (type == 'npy') {
        return npyjs.parse(await res.arrayBuffer())
      } else if (type == 'json') {
        return await res.json()
      } else if (type == 'jsonl') {
        var text = await res.text()
        return text.split(/\r?\n/).filter(d => d).map(line => JSON.parse(line))
      } else if (type == 'json.gz' || type == 'gz' && path.endsWith('.json.gz')) {
        var compressedData = await res.arrayBuffer()
        var decompressed = pako.inflate(new Uint8Array(compressedData), { to: 'string' })
        return JSON.parse(decompressed)
      } else if (type == 'bin') {
        var bytes = new Uint8Array(await res.arrayBuffer())
        var dataLength = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)
        var decompressed = pako.inflate(bytes.slice(4, 4 + dataLength), { to: 'string' })
        return JSON.parse(decompressed)
      } else {
        return await res.text()
      }
    }
  }
  
  
  function addAxisLabel(c, xText, yText, title='', xOffset=0, yOffset=0, titleOffset=0){
    c.svg.select('.x').append('g')
      .translate([c.width/2, xOffset + 25])
      .append('text.axis-label')
      .text(xText)
      .at({textAnchor: 'middle', fill: '#000'})

    c.svg.select('.y')
      .append('g')
      .translate([yOffset -30, c.height/2])
      .append('text.axis-label')
      .text(yText)
      .at({textAnchor: 'middle', fill: '#000', transform: 'rotate(-90)'})

    c.svg
      .append('g.axis').at({fontFamily: 'sans-serif'})
      .translate([c.width/2, titleOffset -10])
      .append('text.axis-label.axis-title')
      .text(title)
      .at({textAnchor: 'middle', fill: '#000'})
  }

  function ggPlot(c){
    c.svg.append('rect.bg-rect')
      .at({width: c.width, height: c.height, fill: c.isBlack ? '#000' : '#EAECED'}).lower()
    c.svg.selectAll('.domain').remove()

    c.svg.selectAll('.x text').at({y: 4})
    c.svg.selectAll('.x .tick')
      .selectAppend('path').at({d: 'M 0 0 V -' + c.height, stroke: c.isBlack ? '#444' : '#fff', strokeWidth: 1})

    c.svg.selectAll('.y text').at({x: -3})
    c.svg.selectAll('.y .tick')
      .selectAppend('path').at({d: 'M 0 0 H ' + c.width, stroke: c.isBlack? '#444' : '#fff', strokeWidth: 1})

    ggPlotUpdate(c)
  }

  function ggPlotUpdate(c){
    c.svg.selectAll('.tick').selectAll('line').remove()

    c.svg.selectAll('.x text').at({y: 4})
    c.svg.selectAll('.x .tick')
      .selectAppend('path').at({d: 'M 0 0 V -' + c.height, stroke: c.isBlack ? '#444' : '#fff', strokeWidth: 1})

    c.svg.selectAll('.y text').at({x: -3})
    c.svg.selectAll('.y .tick')
      .selectAppend('path').at({d: 'M 0 0 H ' + c.width, stroke: c.isBlack? '#444' : '#fff', strokeWidth: 1})
  }

  function initRenderAll(fnLabels){
    var rv = {}
    fnLabels.forEach(label => {
      rv[label] = (ev) => Object.values(rv[label].fns).forEach(d => d(ev))
      rv[label].fns = []
    })

    return rv
  }
  
  function attachRenderAllHistory(renderAll, skipKeys=['hoverId', 'hoverIdx']) {
    // Add state pushing to each render function
    Object.keys(renderAll).forEach(key => {
      renderAll[key].fns.push(() => {
        if (skipKeys.includes(key)) return
        var simpleVisState = {...visState}
        skipKeys.forEach(key => delete simpleVisState[key])

        var url = new URL(window.location) 
        if (visState[key] == url.searchParams.get(key)) return
        url.searchParams.set(key, simpleVisState[key])
        history.pushState(simpleVisState, '', url)
      })
    })

    // Handle back/forward navigation
    d3.select(window).on('popstate.updateState', ev => {
      if (!ev.state) return
      ev.preventDefault()
      Object.keys(renderAll).forEach(key => {
        if (skipKeys.includes(key)) return 
        if (visState[key] == ev.state[key]) return
        visState[key] = ev.state[key]
        renderAll[key]()
      })
    })
  }
  
  function throttle(fn, delay){
    var lastCall = 0
    return (...args) => {
      if (Date.now() - lastCall < delay) return
      lastCall = Date.now()
      fn(...args)
    }
  }

  function debounce(fn, delay) {
    var timeout
    return (...args) => {
      clearTimeout(timeout)
      timeout = setTimeout(() => fn(...args), delay)
    }
  }

  function throttleDebounce(fn, delay) {
    var lastCall = 0
    var timeoutId

    return function (...args) {
      clearTimeout(timeoutId)
      var remainingDelay = delay - (Date.now() - lastCall)
      if (remainingDelay <= 0) {
        lastCall = Date.now()
        fn.apply(this, args)
      } else {
        timeoutId = setTimeout(() => {
          lastCall = Date.now()
          fn.apply(this, args)
        }, remainingDelay)
      }
    }
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  function cantorUnpair(z) {
    const w = Math.floor((Math.sqrt(8 * z + 1) - 1) / 2)
    const t = (w * w + w) / 2
    const y = z - t
    const x = w - y
    return [x, y]
  }
  
  function cache(fn){
    var cache = {}
    return function(...args){
      var key = JSON.stringify(args)
      if (!(key in cache)) cache[key] = fn.apply(this, args)
      return cache[key]
    }
  }
  
  async function initGraphSelect(sel, cgSlug){
    var {graphs} = await util.getFile('./data/graph-metadata.json')
    
    var selectSel = sel.html('').append('select.graph-prompt-select')
      .on('change', function() {
        cgSlug = this.value 
        // visState.clickedId = undefined
        util.params.set('slug', this.value)
        render()
      })
    
    var cgSel = sel.append('div.cg-container')
  
    selectSel.appendMany('option', graphs)
      .text(d => {
        var scanName = util.nameToPrettyPrint[d.scan] || d.scan
        var prefix = d.title_prefix ? d.title_prefix + ' ' : ''
        return prefix + scanName + ' — ' + d.prompt
      })
      .attr('value', d => d.slug)
      .property('selected', d => d.slug === cgSlug)
  
    function render() {
      initCg(cgSel.html(''), cgSlug, {
        isModal: true,
        // clickedId: visState.clickedId,
        // clickedIdCb: id => util.params.set('clickedId', id)
      })
      
      var m = graphs.find(g => g.slug == cgSlug)
      if (!m) return
      selectSel.at({title: m.prompt})
    }
    render()
  }
  
  function attachCgLinkEvents(sel, cgSlug, figmaSlug){
    sel
      .on('mouseover', () => util.getFile(`./graph_data/${cgSlug}.json`))
      .on('click', (ev) => {
        ev.preventDefault()
        
        if (window.innerWidth < 900 || window.innerHeight < 500) {
          return window.open(`./static_js/attribution_graphs/index.html?slug=${cgSlug}`, '_blank')
        }
  
        d3.select('body').classed('modal-open', true)
        var contentSel = d3.select('modal').classed('is-active', 1)
          .select('.modal-content').html('')
        
        util.initGraphSelect(contentSel, cgSlug)
        
        util.params.set('slug', cgSlug)
        if (figmaSlug) history.replaceState(null, '', '#' + figmaSlug)
      })
  }
  
  // TODO: tidy
  function ppToken(d){
    return d
  }
  
  function ppClerp(d){
    return d
  }
  
  
  var scanSlugToName = {
    'h35': 'jackl-circuits-runs-1-4-sofa-v3_0',
    '18l': 'jackl-circuits-runs-1-1-druid-cp_0',
    'moc': 'jackl-circuits-runs-12-19-valet-m_0'
  }
  
  var nameToPrettyPrint = {
    'jackl-circuits-runs-1-4-sofa-v3_0': 'Haiku',
    'jackl-circuits-runs-1-1-druid-cp_0': '18L',
    'jackl-circuits-runs-12-19-valet-m_0': 'Model Organism',
    'jackl-circuits-runs-1-12-rune-cp3_0': '18L PLTs',
  }

  
  return {
    scanSlugToName,
    nameToPrettyPrint,
    params,
    getFile,
    addAxisLabel,
    ggPlot,
    ggPlotUpdate,
    initRenderAll,
    attachRenderAllHistory,
    throttle,
    debounce,
    throttleDebounce,
    sleep,
    cache,
    initGraphSelect,
    attachCgLinkEvents,
    ppToken,
    ppClerp,
    cantorUnpair,
  }
})()

window.init?.()
