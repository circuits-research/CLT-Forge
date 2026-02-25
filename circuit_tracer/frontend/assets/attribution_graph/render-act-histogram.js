/**
 * Render activation histogram for the selected feature.
 *
 * @param {Object} params - Parameters for rendering.
 * @param {d3.Selection} params.featureTitleSel - Selection of the feature title element.
 * @param {string} params.scan - Scan identifier used to load the feature data.
 * @param {Object} params.featureNode - The feature node object containing feature data.
 * @param {Object} params.featureExamples - The feature examples object used to load feature data.
 */

// Histogram size constants
const HISTOGRAM_CONTAINER_WIDTH = 240;
const HISTOGRAM_CONTAINER_HEIGHT = 160;
const HISTOGRAM_SVG_WIDTH = 220;
const HISTOGRAM_SVG_HEIGHT = 140;

window.renderActHistogram = function({ featureTitleSel, scan, featureNode, featureExamples }) {
  const dataPromise = featureExamples.loadFeature(scan, featureNode.featureIndex)
  const placeholderSel = featureTitleSel
    .on('mouseenter', function () {
      d3.select(this).select('.histogram-placeholder').style('display', 'block');
    })
    .on('mouseleave', function () {
      d3.select(this).select('.histogram-placeholder').style('display', 'none');
    })
    .append('div.histogram-placeholder')
    .style('display', 'none')
    .style('position', 'absolute')
    .style('width', `${HISTOGRAM_CONTAINER_WIDTH}px`)
    .style('height', `${HISTOGRAM_CONTAINER_HEIGHT}px`)
    .style('background-color', '#fff')
    .style('border-radius', '8px')
    .style('border', '1px solid #ddd')
    .style('top', `-${HISTOGRAM_CONTAINER_HEIGHT - 40}px`)
    .style('left', '120px')
    .style('z-index', '1000000')
    .style('box-shadow', '0 3px 10px rgba(0,0,0,0.15)')
    .style('padding', '10px');

  const svg = placeholderSel.append('svg')
    .style('width', '100%')
    .style('height', '100%');

  dataPromise.then(featureData => {
    let quantileInfo = '';
    if (featureData && featureData.quantile_values) {
      const quantileValues = featureData.quantile_values;
      let quantile = 0;
      for (let i = 0; i < quantileValues.length; i++) {
        if (featureNode.activation <= quantileValues[i]) {
          quantile = i + 1;
          break;
        }
      }
      if (quantile === 0 && featureNode.activation > quantileValues[quantileValues.length - 1]) {
        quantile = 100;
      }
      quantileInfo = ` ${quantile}pct`;
    }
    featureTitleSel.select('span').text(
      `Act: ${featureNode.activation.toFixed(2)}${quantileInfo}${featureData.activation_frequency !== undefined ? `, Freq: ${(featureData.activation_frequency * 100).toFixed(3)}%` : ''}`
    );
  });

  dataPromise
    .then(featureData => {
      if (!(featureData && featureData.histogram)) return;
      drawHistogram(svg, featureData.histogram, featureNode.activation, featureData.act_min || 0, featureData.act_max || 1);
    })
    .catch(error => {
      console.error('Error loading histogram data:', error);
    });
};

function drawHistogram(svg, histogramData, currentActivation, actMin, actMax) {
  const width = HISTOGRAM_SVG_WIDTH;
  const height = HISTOGRAM_SVG_HEIGHT;
  const margin = { top: 10, right: 10, bottom: 30, left: 10 };

  const maxValue = Math.max(...histogramData);

  const xScale = d3.scaleLinear()
    .domain([0, histogramData.length - 1])
    .range([margin.left, width - margin.right]);

  const yScale = d3.scaleLinear()
    .domain([0, maxValue])
    .range([height - margin.bottom, margin.top]);

  const binPosition = Math.max(
    0,
    Math.min(
      histogramData.length - 1,
      Math.floor(((currentActivation - actMin) / (actMax - actMin)) * histogramData.length)
    )
  );

  svg.html('');

  svg.append('text')
    .attr('x', width / 2)
    .attr('y', 8)
    .attr('text-anchor', 'middle')
    .style('font-size', '10px')
    .style('font-weight', 'bold')
    .text('Activation Distribution');

  svg.selectAll('.bar')
    .data(histogramData)
    .enter()
    .append('rect')
    .attr('class', 'bar')
    .attr('x', (d, i) => xScale(i))
    .attr('width', width / histogramData.length)
    .attr('y', d => yScale(d))
    .attr('height', d => height - margin.bottom - yScale(d))
    .attr('fill', '#e67e22')
    .attr('opacity', 0.8)
    .attr('shape-rendering', 'crispEdges');

  svg.append('line')
    .attr('x1', xScale(binPosition))
    .attr('y1', margin.top)
    .attr('x2', xScale(binPosition))
    .attr('y2', height - margin.bottom)
    .attr('stroke', '#2980b9')
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '4,2');

  svg.append('text')
    .attr('x', xScale(binPosition))
    .attr('y', margin.top + 10)
    .attr('text-anchor', 'middle')
    .style('font-size', '8px')
    .style('font-weight', 'bold')
    .style('fill', '#2980b9')
    .text('Current');

  svg.append('line')
    .attr('x1', margin.left)
    .attr('y1', height - margin.bottom)
    .attr('x2', width - margin.right)
    .attr('y2', height - margin.bottom)
    .attr('stroke', '#777')
    .attr('stroke-width', 1);

  svg.append('line')
    .attr('x1', margin.left)
    .attr('y1', margin.top)
    .attr('x2', margin.left)
    .attr('y2', height - margin.bottom)
    .attr('stroke', '#777')
    .attr('stroke-width', 1);

  svg.append('text')
    .attr('x', margin.left)
    .attr('y', height - margin.bottom + 15)
    .attr('text-anchor', 'middle')
    .style('font-size', '9px')
    .text(actMin.toFixed(2));

  svg.append('text')
    .attr('x', width - margin.right)
    .attr('y', height - margin.bottom + 15)
    .attr('text-anchor', 'middle')
    .style('font-size', '9px')
    .text(actMax.toFixed(2));

  const numTicks = 5;
  for (let i = 0; i < numTicks; i++) {
    const xPos = margin.left + (i * (width - margin.left - margin.right) / (numTicks - 1));
    svg.append('line')
      .attr('x1', xPos)
      .attr('y1', height - margin.bottom)
      .attr('x2', xPos)
      .attr('y2', height - margin.bottom + 3)
      .attr('stroke', '#777')
      .attr('stroke-width', 1);
  }

  svg.append('text')
    .attr('x', width / 2)
    .attr('y', height - 5)
    .attr('text-anchor', 'middle')
    .style('font-size', '8px')
    .text('Activation Value');
}
