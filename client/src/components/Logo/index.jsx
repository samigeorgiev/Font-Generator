import React from 'react';

import FGLogo from 'assets/images/FGLogo.png';
import GitHubLogo from 'assets/images/GitHubLogo.png';

import styles from './index.module.css';

const logo = props => {
    const images = {
        'FGLogo': FGLogo,
        'GitHubLogo': GitHubLogo
    };

    return (
        <div className={styles.Logo}>
            <img src={images[props.src]} alt="FGLogo"/>
        </div>
    );
};

export default logo;