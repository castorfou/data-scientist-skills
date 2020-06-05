import classNames from 'classnames';
import React from 'react';
import Utils from '../../common/utils/Utils';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import { DEFAULT_EXPANDED_VALUE } from './ExperimentView';
import { CollapsibleTagsCell } from './CollapsibleTagsCell';
import _ from 'lodash';

export default class ExperimentViewUtil {
  /** Returns checkbox cell for a row. */
  static getCheckboxForRow(selected, checkboxHandler, cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent key='meta-check' className='run-table-container'>
        <div>
          <input type='checkbox' checked={selected} onClick={checkboxHandler} />
        </div>
      </CellComponent>
    );
  }

  static styles = {
    sortIconStyle: {
      verticalAlign: 'middle',
      fontSize: 20,
    },
    headerCellText: {
      verticalAlign: 'middle',
    },
    sortIconContainer: {
      marginLeft: 2,
      minWidth: 12.5,
      display: 'inline-block',
    },
    expander: {
      pointer: 'cursor',
    },
    runInfoCell: {
      maxWidth: 250,
    },
  };

  /**
   * Returns an icon depending on run status.
   */
  static getRunStatusIcon(status) {
    switch (status) {
      case 'FAILED':
      case 'KILLED':
        return <i className='far fa-times-circle' style={{ color: '#DB1905' }} />;
      case 'FINISHED':
        return <i className='far fa-check-circle' style={{ color: '#10B36B' }} />;
      case 'SCHEDULED':
        return <i className='far fa-clock' style={{ color: '#258BD2' }} />;
      default:
        return <i />;
    }
  }

  /**
   * Returns table cells describing run metadata (i.e. not params/metrics) comprising part of
   * the display row for a run.
   */
  static getRunInfoCellsForRow(runInfo, tags, isParent, cellType, handleCellToggle, excludedKeys) {
    const CellComponent = `${cellType}`;
    const user = Utils.getUser(runInfo, tags);
    const queryParams = window.location && window.location.search ? window.location.search : '';
    const sourceType = Utils.renderSource(tags, queryParams);
    const startTime = runInfo.start_time;
    const status = runInfo.status;
    const runName = Utils.getRunName(tags);
    const childLeftMargin = isParent ? {} : { paddingLeft: 16 };
    const columnProps = [
      {
        key: 'status',
        className: 'run-table-container',
        title: status,
        children: ExperimentViewUtil.getRunStatusIcon(status),
      },
      {
        key: ExperimentViewUtil.AttributeColumnLabels.DATE,
        className: 'run-table-container',
        style: { whiteSpace: 'inherit' },
        children: (
          <div style={childLeftMargin}>
            <Link to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}>
              {Utils.formatTimestamp(startTime)}
            </Link>
          </div>
        ),
      },
      {
        key: ExperimentViewUtil.AttributeColumnLabels.USER,
        className: 'run-table-container',
        title: user,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {user}
          </div>
        ),
      },
      {
        key: ExperimentViewUtil.AttributeColumnLabels.RUN_NAME,
        className: 'run-table-container',
        title: runName,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {runName}
          </div>
        ),
      },
      {
        key: ExperimentViewUtil.AttributeColumnLabels.SOURCE,
        className: 'run-table-container',
        title: sourceType,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {Utils.renderSourceTypeIcon(Utils.getSourceType(tags))}
            {sourceType}
          </div>
        ),
      },
      {
        key: ExperimentViewUtil.AttributeColumnLabels.VERSION,
        className: 'run-table-container',
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {Utils.renderVersion(tags)}
          </div>
        ),
      },
      {
        key: 'Tags',
        className: 'run-table-container',
        children: (
          <div style={ExperimentViewUtil.styles.runInfoCell}>
            <CollapsibleTagsCell tags={tags} onToggle={handleCellToggle} />
          </div>
        ),
      },
    ];
    const excludedKeysSet = new Set(excludedKeys);
    return columnProps
      .filter((column) => !excludedKeysSet.has(column.key))
      .map((props) => <CellComponent {...props} />);
  }

  /**
   * Returns an icon for sorting the metric or param column with the specified key. The icon
   * is visible if we're currently sorting by the corresponding column. Otherwise, the icon is
   * invisible but takes up space.
   */
  static getSortIcon(curOrderByKey, curOrderByAsc, canonicalKey) {
    if (curOrderByKey === canonicalKey) {
      return (
        <span>
          <i
            className={curOrderByAsc ? 'fas fa-caret-up' : 'fas fa-caret-down'}
            style={ExperimentViewUtil.styles.sortIconStyle}
          />
        </span>
      );
    }
    return undefined;
  }

  /** Returns checkbox element for selecting all runs */
  static getSelectAllCheckbox(onCheckAll, isAllCheckedBool, cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent key='meta-check' className='bottom-row run-table-container'>
        <input type='checkbox' onChange={onCheckAll} checked={isAllCheckedBool} />
      </CellComponent>
    );
  }

  static AttributeColumnLabels = {
    DATE: 'Start Time',
    USER: 'User',
    RUN_NAME: 'Run Name',
    SOURCE: 'Source',
    VERSION: 'Version',
  };

  /**
   * Returns header-row table cells for columns containing run metadata.
   */
  static getRunMetadataHeaderCells(onSortBy, curOrderByKey, curOrderByAsc, cellType, excludedCols) {
    const CellComponent = `${cellType}`;
    const getHeaderCell = (key, text, canonicalSortKey) => {
      const sortIcon = ExperimentViewUtil.getSortIcon(
        curOrderByKey,
        curOrderByAsc,
        canonicalSortKey,
      );
      const isSortable = canonicalSortKey !== null;
      const cellClassName = classNames('bottom-row', 'run-table-container', {
        sortable: isSortable,
      });
      return (
        <CellComponent
          key={'meta-' + key}
          className={cellClassName}
          onClick={() => (isSortable ? onSortBy(canonicalSortKey, !curOrderByAsc) : null)}
        >
          <span style={ExperimentViewUtil.styles.headerCellText}>{text}</span>
          {isSortable && (
            <span style={ExperimentViewUtil.styles.sortIconContainer}>{sortIcon}</span>
          )}
        </CellComponent>
      );
    };
    const excludedColsSet = new Set(excludedCols);
    return [
      {
        key: 'status',
        canonicalSortKey: null,
      },
      {
        key: 'start_time',
        displayName: this.AttributeColumnLabels.DATE,
        canonicalSortKey: 'attributes.start_time',
      },
      {
        key: 'user_id',
        displayName: this.AttributeColumnLabels.USER,
        canonicalSortKey: 'tags.`mlflow.user`',
      },
      {
        key: 'run_name',
        displayName: this.AttributeColumnLabels.RUN_NAME,
        canonicalSortKey: 'tags.`mlflow.runName`',
      },
      {
        key: 'source',
        displayName: this.AttributeColumnLabels.SOURCE,
        canonicalSortKey: 'tags.`mlflow.source.name`',
      },
      {
        key: 'source_version',
        displayName: this.AttributeColumnLabels.VERSION,
        canonicalSortKey: 'tags.`mlflow.source.git.commit`',
      },
      {
        key: 'tags',
        displayName: 'Tags',
        canonicalSortKey: null,
      },
    ]
      .filter((column) => !excludedColsSet.has(column.displayName))
      .map((h) => getHeaderCell(h.key, <span>{h.displayName}</span>, h.canonicalSortKey));
  }

  static makeCanonicalKey(keyType, keyName) {
    return keyType + '.`' + keyName + '`';
  }

  static getExpanderHeader(cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent
        key={'meta-expander'}
        className={'bottom-row run-table-container'}
        style={{ width: '5px' }}
      />
    );
  }

  /**
   * Returns a table cell corresponding to a single metric value. The metric is assumed to be
   * unbagged (marked to be displayed in its own column).
   * @param metricKey The key of the desired metric
   * @param metricsMap Object mapping metric keys to their latest values for a single run
   * @param metricRanges Object mapping metric keys to objects of the form {min: ..., max: ...}
   *                     containing min and max values of the metric across all visible runs.
   * @param cellType Tag type (string like "div", "td", etc) of containing cell.
   */
  static getUnbaggedMetricCell(metricKey, metricsMap, metricRanges, cellType) {
    const className = 'left-border run-table-container';
    const keyName = 'metric-' + metricKey;
    const CellComponent = `${cellType}`;
    if (metricsMap[metricKey]) {
      const metric = metricsMap[metricKey].getValue();
      const range = metricRanges[metricKey];
      let fraction = 1.0;
      if (range.max > range.min) {
        fraction = (metric - range.min) / (range.max - range.min);
      }
      const percent = fraction * 100 + '%';
      return (
        <CellComponent className={className} key={keyName}>
          {/* We need the extra div because metric-filler-bg is inline-block */}
          <div>
            <div className='metric-filler-bg'>
              <div className='metric-filler-fg' style={{ width: percent }} />
              <div className='metric-text'>{Utils.formatMetric(metric)}</div>
            </div>
          </div>
        </CellComponent>
      );
    }
    return <CellComponent className={className} key={keyName} />;
  }

  static getUnbaggedParamCell(paramKey, paramsMap, cellType) {
    const CellComponent = `${cellType}`;
    const className = 'left-border run-table-container';
    const keyName = 'param-' + paramKey;
    if (paramsMap[paramKey]) {
      return (
        <CellComponent className={className} key={keyName}>
          <div>{paramsMap[paramKey].getValue()}</div>
        </CellComponent>
      );
    } else {
      return <CellComponent className={className} key={keyName} />;
    }
  }

  static computeMetricRanges(metricsByRun) {
    const ret = {};
    metricsByRun.forEach((metrics) => {
      metrics.forEach((metric) => {
        if (!ret.hasOwnProperty(metric.key)) {
          ret[metric.key] = { min: Math.min(metric.value, metric.value * 0.7), max: metric.value };
        } else {
          if (metric.value < ret[metric.key].min) {
            ret[metric.key].min = Math.min(metric.value, metric.value * 0.7);
          }
          if (metric.value > ret[metric.key].max) {
            ret[metric.key].max = metric.value;
          }
        }
      });
    });
    return ret;
  }

  /**
   * Turn a list of metrics to a map of metric key to metric.
   */
  static toMetricsMap(metrics) {
    const ret = {};
    metrics.forEach((metric) => {
      ret[metric.key] = metric;
    });
    return ret;
  }

  /**
   * Turn a list of param to a map of metric key to metric.
   */
  static toParamsMap(params) {
    const ret = {};
    params.forEach((param) => {
      ret[param.key] = param;
    });
    return ret;
  }

  static isExpanderOpen(runsExpanded, runId) {
    let expanderOpen = DEFAULT_EXPANDED_VALUE;
    if (runsExpanded[runId] !== undefined) expanderOpen = runsExpanded[runId];
    return expanderOpen;
  }

  static getExpander(hasExpander, expanderOpen, onExpandBound, runUuid, cellType) {
    const CellComponent = `${cellType}`;
    if (!hasExpander) {
      return <CellComponent key={'Expander-' + runUuid} style={{ padding: 8 }}></CellComponent>;
    }
    if (expanderOpen) {
      return (
        <CellComponent onClick={onExpandBound} key={'Expander-' + runUuid} style={{ padding: 8 }}>
          <i className='ExperimentView-expander far fa-minus-square' />
        </CellComponent>
      );
    } else {
      return (
        <CellComponent onClick={onExpandBound} key={'Expander-' + runUuid} style={{ padding: 8 }}>
          <i className='ExperimentView-expander far fa-plus-square' />
        </CellComponent>
      );
    }
  }

  static getRowRenderMetadata({ runInfos, tagsList, runsExpanded }) {
    const runIdToIdx = {};
    runInfos.forEach((r, idx) => {
      runIdToIdx[r.run_uuid] = idx;
    });
    const treeNodes = runInfos.map((r) => new TreeNode(r.run_uuid));
    tagsList.forEach((tags, idx) => {
      const parentRunId = tags['mlflow.parentRunId'];
      if (parentRunId) {
        const parentRunIdx = runIdToIdx[parentRunId.value];
        if (parentRunIdx !== undefined) {
          treeNodes[idx].parent = treeNodes[parentRunIdx];
        }
      }
    });
    // Map of parentRunIds to list of children runs (idx)
    const parentIdToChildren = {};
    treeNodes.forEach((t, idx) => {
      const root = t.findRoot();
      if (root !== undefined && root.value !== t.value) {
        const old = parentIdToChildren[root.value];
        let newList;
        if (old) {
          old.push(idx);
          newList = old;
        } else {
          newList = [idx];
        }
        parentIdToChildren[root.value] = newList;
      }
    });

    const parentRows = _.flatMap([...Array(runInfos.length).keys()], (idx) => {
      if (treeNodes[idx].isCycle() || !treeNodes[idx].isRoot()) return [];
      const runId = runInfos[idx].run_uuid;
      let hasExpander = false;
      let childrenIds = undefined;
      if (parentIdToChildren[runId]) {
        hasExpander = true;
        childrenIds = parentIdToChildren[runId].map((cIdx) => runInfos[cIdx].run_uuid);
      }
      return [
        {
          idx,
          isParent: true,
          hasExpander,
          expanderOpen: ExperimentViewUtil.isExpanderOpen(runsExpanded, runId),
          childrenIds,
          runId,
        },
      ];
    });
    const mergedRows = [];
    parentRows.forEach((r) => {
      const runId = r.runId;
      mergedRows.push(r);
      const childrenIdxs = parentIdToChildren[runId];
      if (childrenIdxs) {
        if (ExperimentViewUtil.isExpanderOpen(runsExpanded, runId)) {
          const childrenRows = childrenIdxs.map((idx) => {
            return { idx, isParent: false, hasExpander: false };
          });
          mergedRows.push(...childrenRows);
        }
      }
    });
    return mergedRows.slice(0);
  }

  static getRows({ runInfos, tagsList, runsExpanded, getRow }) {
    const mergedRows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
    });
    return mergedRows.map((rowMetadata) => getRow(rowMetadata));
  }

  static renderRows(rows) {
    return rows.map((row) => {
      const style = row.isChild ? { backgroundColor: '#fafafa' } : {};
      return (
        <tr key={row.key} style={style} className='ExperimentView-row'>
          {row.contents}
        </tr>
      );
    });
  }
}

export class TreeNode {
  constructor(value) {
    this.value = value;
    this.parent = undefined;
  }
  /**
   * Returns the root node. If there is a cycle it will return undefined;
   */
  findRoot() {
    const visited = new Set([this.value]);
    let current = this;
    while (current.parent !== undefined) {
      if (visited.has(current.parent.value)) {
        return undefined;
      }
      visited.add(current.value);
      current = current.parent;
    }
    return current;
  }
  isRoot() {
    return this.findRoot().value === this.value;
  }
  isCycle() {
    return this.findRoot() === undefined;
  }
}
