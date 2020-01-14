import React from 'react';

import FGLogo from 'assets/images/FGLogo.png';
import GitHubLogo from 'assets/images/GitHubLogo.png';
import GoogleLogo from 'assets/images/GoogleLogo.png';
import FacebookLogo from 'assets/images/FacebookLogo.png';

import styles from './index.module.css';

const logo = props => {
    const images = {
        'FGLogo': FGLogo,
        'GitHubLogo': GitHubLogo,
        'Google': GoogleLogo,
        'Facebook': FacebookLogo
    };

    return (
        <div className={styles.Logo}>
            <img src={images[props.src]} alt="FGLogo" style={{
                filter: `brightness(${props.brightness})`
            }} />
        </div>
    );
};

export default logo;