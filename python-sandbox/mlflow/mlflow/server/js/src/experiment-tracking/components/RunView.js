import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './RunView.css';
import { HtmlTableView } from './HtmlTableView';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import { Dropdown, MenuItem } from 'react-bootstrap';
import ArtifactPage from './ArtifactPage';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { Experiment } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';
import { NOTE_CONTENT_TAG, NoteInfo } from '../utils/NoteUtils';
import { BreadcrumbTitle } from './BreadcrumbTitle';
import { RenameRunModal } from './modals/RenameRunModal';
import EditableTagsTableView from './EditableTagsTableView';
import { Icon, Descriptions } from 'antd';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { IconButton } from '../../common/components/IconButton';

export class RunViewImpl extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    run: PropTypes.object.isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    experimentId: PropTypes.string.isRequired,
    params: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    latestMetrics: PropTypes.object.isRequired,
    getMetricPagePath: PropTypes.func.isRequired,
    runDisplayName: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    handleSetRunTag: PropTypes.func.isRequired,
  };

  state = {
    showRunRenameModal: false,
    showNoteEditor: false,
    showTags: Utils.getVisibleTagValues(this.props.tags).length > 0,
  };

  componentDidMount() {
    const pageTitle = `${this.props.runDisplayName} - MLflow Run`;
    Utils.updatePageTitle(pageTitle);
  }

  handleRenameRunClick = () => {
    this.setState({ showRunRenameModal: true });
  };

  hideRenameRunModal = () => {
    this.setState({ showRunRenameModal: false });
  };

  getRunCommand() {
    const { tags, params } = this.props;
    let runCommand = null;
    const sourceName = Utils.getSourceName(tags);
    const sourceVersion = Utils.getSourceVersion(tags);
    const entryPointName = Utils.getEntryPointName(tags);
    const backend = Utils.getBackend(tags);
    if (Utils.getSourceType(tags) === 'PROJECT') {
      runCommand = 'mlflow run ' + shellEscape(sourceName);
      if (sourceVersion && sourceVersion !== 'latest') {
        runCommand += ' -v ' + shellEscape(sourceVersion);
      }
      if (entryPointName && entryPointName !== 'main') {
        runCommand += ' -e ' + shellEscape(entryPointName);
      }
      if (backend) {
        runCommand += ' -b ' + shellEscape(backend);
      }
      Object.values(params)
        .sort()
        .forEach((p) => {
          runCommand += ' -P ' + shellEscape(p.key + '=' + p.value);
        });
    }
    return runCommand;
  }

  handleCancelEditNote = () => {
    this.setState({ showNoteEditor: false });
  };

  handleSubmitEditNote = (note) => {
    return this.props.handleSetRunTag(NOTE_CONTENT_TAG, note).then(() => {
      this.setState({ showNoteEditor: false });
    });
  };

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showNoteEditor: true });
  };

  static getRunStatusDisplayName(status) {
    return status !== 'RUNNING' ? status : 'UNFINISHED';
  }

  render() {
    const { runUuid, run, params, tags, latestMetrics, getMetricPagePath } = this.props;
    const { showNoteEditor } = this.state;
    const noteInfo = NoteInfo.fromTags(tags);
    const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
    const duration =
      run.getStartTime() && run.getEndTime() ? run.getEndTime() - run.getStartTime() : null;
    const status = RunViewImpl.getRunStatusDisplayName(run.getStatus());
    const queryParams = window.location && window.location.search ? window.location.search : '';
    const tableStyles = {
      table: {
        width: 'auto',
        minWidth: '400px',
      },
      th: {
        width: 'auto',
        minWidth: '200px',
        marginRight: '80px',
      },
    };
    const runCommand = this.getRunCommand();
    const editIcon = (
      <IconButton
        icon={<Icon className='edit-icon' type='form' />}
        onClick={this.startEditingDescription}
      />
    );
    return (
      <div className='RunView'>
        {/* Breadcrumbs */}
        <div className='header-container'>
          <BreadcrumbTitle experiment={this.props.experiment} title={this.props.runDisplayName} />
          <Dropdown id='dropdown-custom-1' className='mlflow-dropdown'>
            <Dropdown.Toggle noCaret className='mlflow-dropdown-button'>
              <i className='fas fa-caret-down' />
            </Dropdown.Toggle>
            <Dropdown.Menu className='mlflow-menu header-menu'>
              <MenuItem className='mlflow-menu-item' onClick={this.handleRenameRunClick}>
                Rename
              </MenuItem>
            </Dropdown.Menu>
          </Dropdown>
          <RenameRunModal
            runUuid={runUuid}
            onClose={this.hideRenameRunModal}
            runName={this.props.runName}
            isOpen={this.state.showRunRenameModal}
          />
        </div>

        {/* Metadata List */}
        <Descriptions className='metadata-list'>
          <Descriptions.Item label='Date'>{startTime}</Descriptions.Item>
          <Descriptions.Item label='Source'>
            {Utils.renderSourceTypeIcon(Utils.getSourceType(tags))}
            {Utils.renderSource(tags, queryParams)}
          </Descriptions.Item>
          {Utils.getSourceVersion(tags) ? (
            <Descriptions.Item label='Git Commit'>
              {Utils.renderVersion(tags, false)}
            </Descriptions.Item>
          ) : null}
          {Utils.getSourceType(tags) === 'PROJECT' ? (
            <Descriptions.Item label='Entry Point'>
              {Utils.getEntryPointName(tags) || 'main'}
            </Descriptions.Item>
          ) : null}
          <Descriptions.Item label='User'>{Utils.getUser(run, tags)}</Descriptions.Item>
          {duration !== null ? (
            <Descriptions.Item label='Duration'>{Utils.formatDuration(duration)}</Descriptions.Item>
          ) : null}
          <Descriptions.Item label='Status'>{status}</Descriptions.Item>
          {tags['mlflow.parentRunId'] !== undefined ? (
            <Descriptions.Item label='Parent Run'>
              <Link
                to={Routes.getRunPageRoute(
                  this.props.experimentId,
                  tags['mlflow.parentRunId'].value,
                )}
              >
                {tags['mlflow.parentRunId'].value}
              </Link>
            </Descriptions.Item>
          ) : null}
          {tags['mlflow.databricks.runURL'] !== undefined ? (
            <Descriptions.Item label='Job Output'>
              <a
                href={Utils.setQueryParams(tags['mlflow.databricks.runURL'].value, queryParams)}
                target='_blank'
              >
                Logs
              </a>
            </Descriptions.Item>
          ) : null}
        </Descriptions>

        {/* Page Sections */}
        <div className='RunView-info'>
          {runCommand ? (
            <CollapsibleSection title='Run Command'>
              <textarea className='run-command text-area' readOnly value={runCommand} />
            </CollapsibleSection>
          ) : null}
          <CollapsibleSection
            title={<span>Notes {showNoteEditor ? null : editIcon}</span>}
            forceOpen={showNoteEditor}
          >
            <EditableNote
              defaultMarkdown={noteInfo && noteInfo.content}
              onSubmit={this.handleSubmitEditNote}
              onCancel={this.handleCancelEditNote}
              showEditor={showNoteEditor}
            />
          </CollapsibleSection>
          <CollapsibleSection title='Parameters'>
            <HtmlTableView
              columns={['Name', 'Value']}
              values={getParamValues(params)}
              styles={tableStyles}
            />
          </CollapsibleSection>
          <CollapsibleSection title='Metrics'>
            <HtmlTableView
              columns={['Name', 'Value']}
              values={getMetricValues(latestMetrics, getMetricPagePath)}
              styles={tableStyles}
            />
          </CollapsibleSection>
          <CollapsibleSection title='Tags'>
            <EditableTagsTableView runUuid={runUuid} tags={tags} />
          </CollapsibleSection>
          <CollapsibleSection title='Artifacts'>
            <ArtifactPage runUuid={runUuid} />
          </CollapsibleSection>
        </div>
      </div>
    );
  }

  handleSubmittedNote(err) {
    if (err) {
      // Do nothing; error is handled by the note editor view
    } else {
      // Successfully submitted note, close the editor
      this.setState({ showNotesEditor: false });
    }
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuid, experimentId } = ownProps;
  const run = getRunInfo(runUuid, state);
  const experiment = getExperiment(experimentId, state);
  const params = getParams(runUuid, state);
  const tags = getRunTags(runUuid, state);
  const latestMetrics = getLatestMetrics(runUuid, state);
  const runDisplayName = Utils.getRunDisplayName(tags, runUuid);
  const runName = Utils.getRunName(tags, runUuid);
  return { run, experiment, params, tags, latestMetrics, runDisplayName, runName };
};

export const RunView = connect(mapStateToProps)(RunViewImpl);

// Private helper functions.

const getParamValues = (params) => {
  return Object.values(params)
    .sort()
    .map((p) => [p.getKey(), p.getValue()]);
};

const getMetricValues = (latestMetrics, getMetricPagePath) => {
  return Object.values(latestMetrics)
    .sort()
    .map((m) => {
      const key = m.key;
      return [
        <Link to={getMetricPagePath(key)} title='Plot chart'>
          {key}
          <i className='fas fa-chart-line' style={{ paddingLeft: '6px' }} />
        </Link>,
        <span title={m.value}>{Utils.formatMetric(m.value)}</span>,
      ];
    });
};

const shellEscape = (str) => {
  if (/["\r\n\t ]/.test(str)) {
    return '"' + str.replace(/"/g, '\\"') + '"';
  }
  return str;
};
